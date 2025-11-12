import tensorflow as tf


@tf.keras.utils.register_keras_serializable(name="LAN")
class NAC_FC(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        topk: int = 8,
        mode : str = 'exact',           # 'steady', 'euler', or 'exact'
        euler_steps : int = 5,
        sparsity: float = 0.5,
        dt: float = 1.0,
        delta_t: float = 0.5,
        activation=None,
        dropout: float = 0.0,
        tau_epsilon: float = 1e-6,
        use_bias: bool = True,
        return_attention: bool = False,
        return_sequences: bool = False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # Basic checks and parameter setup
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert 0.0 <= sparsity <= 0.9, "sparsity must be in [0.0, 0.9]"

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.depth = self.d_model // self.num_heads
        self.topk = int(topk)
        self.mode = mode
        self.euler_steps = int(euler_steps)
        self.sparsity = float(sparsity)
        self.delta_t = float(delta_t)
        self.dt = float(dt)
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = float(dropout)
        self.tau_epsilon = float(tau_epsilon)
        self.use_bias = bool(use_bias)
        self.return_attention = bool(return_attention)
        self.return_sequences = bool(return_sequences)

        # projections
        self.q_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="q_proj")
        self.k_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="k_proj")
        self.v_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="v_proj")
        self.out_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="out_proj")

        # time layers
        self.time_a = tf.keras.layers.Dense(1, name="time_a")
        self.time_b = tf.keras.layers.Dense(1, name="time_b")

        # phi and tau MLPs
        self.phi_in = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="phi_in")
        self.phi_out = tf.keras.layers.Dense(1, use_bias=True, name="phi_out")

        # make tau smaller MLP to reduce params
        self.tau_in = tf.keras.layers.Dense(max(4, int(self.d_model // 2)), use_bias=self.use_bias, name="tau_in")
        self.tau_out = tf.keras.layers.Dense(1, use_bias=self.use_bias, name="tau_out")

        self.attn_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def split_heads(self, x):
        # x: [B, T, d_model] -> [B, H, T, depth]
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        x = tf.reshape(x, (B, T, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        # x: [B, H, T, depth] -> [B, T, d_model]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        return tf.reshape(x, (B, T, self.d_model))

    def sparse_topk_pairwise(self, q, k, K=None):
        """
        Return:
            topk_pairs: [B, H, Tq, K_eff, 2D]
            topk_idx:   [B, H, Tq, K_eff]
        """
        if K is None:
            K = self.topk

        Tk = tf.shape(k)[2]

        # effective K to prevent top_k error on short sequences
        K_eff = tf.minimum(K, Tk)

        # similarity scores: [B,H,Tq,Tk]
        scores = tf.einsum("bhqd,bhkd->bhqk", q, k)

        # top-k along Tk dimension (safe because K_eff <= Tk)
        _, topk_idx = tf.math.top_k(scores, k=K_eff)

        # gather top-k keys: batch_dims=2 handles B,H
        # k: [B,H,Tk,D], topk_idx: [B,H,Tq,K_eff] -> gathered: [B,H,Tq,K_eff,D]
        k_gathered = tf.gather(k, topk_idx, batch_dims=2, axis=2)

        # tile q to match K dim: q [B,H,Tq,D] -> [B,H,Tq,K_eff,D]
        q_tiled = tf.tile(tf.expand_dims(q, 3), [1, 1, 1, K_eff, 1])

        # concat q and gathered k -> [B,H,Tq,K_eff,2D]
        topk_pairs = tf.concat([q_tiled, k_gathered], axis=-1)

        return topk_pairs, topk_idx

    def compute_phi_tau(self, q, k, t):
        """
        Uses sparse top-K pairs and returns:
            phi: [B,H,Tq,K_eff]
            tau: [B,H,Tq,K_eff]
            t_interp: [B,H,Tq,K_eff]
            topk_idx: [B,H,Tq,K_eff]
        """
        B = tf.shape(q)[0]
        H = tf.shape(q)[1]
        Tq = tf.shape(q)[2]

        pair, topk_idx = self.sparse_topk_pairwise(q, k, K=self.topk)
        K_eff = tf.shape(pair)[3]
        D2 = tf.shape(pair)[-1]

        # flatten for MLP: [B*H*Tq*K_eff, 2D]
        flat_pair = tf.reshape(pair, [-1, D2])

        # phi
        x = self.phi_in(flat_pair)
        phi_raw = self.phi_out(x)  # [N,1]
        phi = tf.reshape(tf.nn.sigmoid(phi_raw), [B, H, Tq, K_eff])

        # time interpolation
        t_a = self.time_a(flat_pair)
        t_b = self.time_b(flat_pair)
        t_scalar = tf.cast(t, flat_pair.dtype)
        t_interp_flat = tf.nn.sigmoid(t_a * t_scalar + t_b)
        t_interp = tf.reshape(t_interp_flat, [B, H, Tq, K_eff])

        # tau (positive)
        tau_x = self.tau_in(flat_pair)
        tau_raw = self.tau_out(tau_x)
        tau = tf.reshape(tf.nn.softplus(tau_raw) + self.tau_epsilon, [B, H, Tq, K_eff])

        return phi, tau, t_interp, topk_idx

    def call(self, inputs, mask=None, training=None):
        # parse inputs: support single tensor or (q,k,v) etc.
        t = tf.constant(1.0, dtype=tf.float32)
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, t = inputs
            q_in = k_in = v_in = x
        elif isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            q_in, k_in, v_in = inputs
        elif isinstance(inputs, (list, tuple)) and len(inputs) == 4:
            q_in, k_in, v_in, t = inputs
            t = tf.cast(t, tf.float32)
        else:
            q_in = k_in = v_in = inputs

        # linear projections
        q = self.q_dense(q_in)  # [B,T,d_model]
        k = self.k_dense(k_in)
        v = self.v_dense(v_in)

        # split heads: [B,H,T,depth]
        qh = self.split_heads(q)
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        # compute sparse phi/tau/time and indices
        phi, tau, time_interp, topk_idx = self.compute_phi_tau(qh, kh, t)

        # Solve dynamics based on chosen mode
        if self.mode == 'steady':
            attn_logits = phi / tau

        elif self.mode == 'exact':
            attn_logits = (phi / tau) * (1 - tf.exp(-tau * time_interp))

        elif self.mode == 'euler':
            a = tf.zeros_like(phi)
            for _ in range(self.euler_steps):
                increment = self.delta_t * (-tau * a + phi)
                a = a + increment
            attn_logits = a


        # optional mask handling: if mask is provided as [B, T] we must map it to queries/keys.
        if mask is not None:
            mask = tf.cast(mask, attn_logits.dtype)
            mask_exp = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            mask_gathered = tf.gather(mask_exp, topk_idx, batch_dims=2, axis=2)
            attn_logits = attn_logits * mask_gathered

        # normalize over K dimension
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)  # [B,H,Tq,K_eff]

        # gather top-K values corresponding to topk_idx: vh [B,H,T,D]
        # gathered shape: [B,H,Tq,K_eff,D]
        vh_topk = tf.gather(vh, topk_idx, batch_dims=2, axis=2)


        # integration
        attn_weights_exp = tf.expand_dims(attn_weights, axis=-1)  # [B,H,Tq,K,1]
        weighted_values = attn_weights_exp * vh_topk  # [B,H,Tq,K,D]
        output = tf.reduce_sum(weighted_values, axis=3) * self.dt  # [B,H,Tq,D]

        # combine heads -> [B,Tq,d_model]
        combined = self.combine_heads(output)
        out = self.out_dense(combined)
        if self.activation is not None:
            out = self.activation(out)

        result = out if self.return_sequences else out[:, -1, :]
        if self.return_attention:
            return result, attn_weights
        return result

    def get_config(self):
        '''
        Return configuration dictionary for model serialization.
        '''
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "topk": self.topk,
                "mode": self.mode,
                "euler_steps": self.euler_steps,
                "sparsity": self.sparsity,
                "delta_t": self.delta_t,
                "dt": self.dt,
                "activation": tf.keras.activations.serialize(self.activation),
                "dropout": self.dropout_rate,
                "tau_epsilon": self.tau_epsilon,
                "return_attention": self.return_attention,
                "return_sequences": self.return_sequences,
            }
        )
        return config

'''
The code is taken from https://github.com/itxwaleedrazzaq/liquid_attention 
'''
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(name="LAN")
class LAN(tf.keras.layers.Layer):
    """
    Liquid Attention layer (multi-head) with three possible modes:
      - mode='steady' : steady-state solution e* = phi / tau
      - mode='euler'  : forward Euler iteration e_{n+1} = e_n + (dt)(-e_n*tau + phi)
      - mode='exact'  : closed-form solution a(t) = (phi/tau) * (1 - exp(-tau * t))
    """

    def __init__(
        self,
        d_model : int,
        num_heads : int,
        mode : str = 'exact',           # 'steady', 'euler', or 'exact'
        euler_steps : int = 5,
        activation = None,
        phi_hidden : int = 64,
        tau_epsilon : float = 1e-6,
        delta_t : float = 0.5,
        dropout : float = 0.0,
        use_bias : bool = True,
        return_attention : bool = False,
        return_sequences : bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        '''
        Initialize key parameters and sub-layers:
          - d_model: dimension of embeddings
          - num_heads: number of attention heads
          - phi_hidden: hidden units for phi MLP
          - tau_epsilon: small constant to avoid division by zero
          - mode: solution strategy ('steady', 'euler', 'exact')
          - euler_steps: number of iterations for Euler method
          - delta_t: time-step for Euler solver
          - dropout: dropout rate for attention weights
          - activation: activation function for output
          - return_attention: whether to return attention weights
          - return_sequences: whether to return the full sequence or last step
        '''

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert mode in ('steady', 'euler', 'exact'), "mode must be 'steady', 'euler' or 'exact'"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.phi_hidden = phi_hidden
        self.tau_epsilon = tau_epsilon
        self.mode = mode
        self.euler_steps = int(euler_steps)
        self.delta_t = float(delta_t)
        self.dropout = dropout
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.return_sequences = return_sequences

        '''
        Define projection layers for queries, keys, values, and output.
        Each projects input into a d_model-dimensional space.
        '''
        self.q_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='q_proj')
        self.k_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='k_proj')
        self.v_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='v_proj')
        self.out_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='out_proj')

        '''
        Define time-dependent layers for interpolation.
        These help model continuous dynamics in attention.
        '''
        self.time_a = tf.keras.layers.Dense(1, name="time_a")
        self.time_b = tf.keras.layers.Dense(1, name="time_b")

        '''
        Define layers for phi and tau computation.
          - phi: computed from concatenated [q; k] through two-layer MLP.
          - tau: positive time constant per pair, enforced via softplus.
        '''
        self.phi_in = tf.keras.layers.Dense(phi_hidden, use_bias=use_bias, name='phi_in')
        self.phi_out = tf.keras.layers.Dense(1, use_bias=True, name='phi_out')

        self.tau_in = tf.keras.layers.Dense(int(phi_hidden/2.0), use_bias=self.use_bias, name='tau_in')
        self.tau_out = tf.keras.layers.Dense(1, use_bias=self.use_bias, name='tau_out')

        self.attn_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        '''Nothing custom to build beyond defined sub-layers.'''
        super().build(input_shape)

    def split_heads(self, x):
        '''
        Reshape input tensor into multiple attention heads.

        Args:
          x: [B, T, d_model]
        Returns:
          [B, num_heads, T, depth]
        '''
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        x = tf.reshape(x, (B, T, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        '''
        Reverse split_heads: combine multi-head output.

        Args:
          x: [B, num_heads, T, depth]
        Returns:
          [B, T, d_model]
        '''
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        return tf.reshape(x, (B, T, self.d_model))

    def pairwise_concat(self, q, k):
        '''
        Concatenate each query with each key across sequence positions.

        Args:
          q: [B, H, Tq, D]
          k: [B, H, Tk, D]
        Returns:
          [B, H, Tq, Tk, 2D]
        '''
        q_exp = tf.expand_dims(q, axis=3)
        k_exp = tf.expand_dims(k, axis=2)
        q_tile = tf.tile(q_exp, [1, 1, 1, tf.shape(k)[2], 1])
        k_tile = tf.tile(k_exp, [1, 1, tf.shape(q)[2], 1, 1])
        return tf.concat([q_tile, k_tile], axis=-1)

    def compute_phi_tau(self, q, k, t):
        '''
        Compute phi (target-content gate), tau (time constant gate) and t (interpolation time) for each query-key pair.

        Args:
          q, k: [B, H, T, D]
          t: scalar or tensor controlling interpolation
        Returns:
          phi: [B, H, Tq, Tk]
          tau: [B, H, Tq, Tk]
          t_interp: interpolated time scaling
        '''
        pair = self.pairwise_concat(q, k)
        _, H, Tq, Tk, _ = tf.shape(pair)[0], tf.shape(pair)[1], tf.shape(pair)[2], tf.shape(pair)[3], tf.shape(pair)[4]

        flat_pair = tf.reshape(pair, (-1, tf.shape(pair)[-1]))

        # Compute phi
        x = self.phi_in(flat_pair)
        phi_raw = self.phi_out(x)
        phi = tf.reshape(phi_raw, (-1, H, Tq, Tk))
        phi = tf.nn.sigmoid(phi)

        # Time interpolation
        t_a = self.time_a(flat_pair)
        t_b = self.time_b(flat_pair)
        t_interp = tf.nn.sigmoid(t_a * t + t_b)
        t_interp = tf.reshape(t_interp, (-1, H, Tq, Tk))

        # Compute tau
        tau_x = self.tau_in(flat_pair)
        tau_raw = self.tau_out(tau_x)
        tau = tf.reshape(tau_raw, (-1, H, Tq, Tk))
        tau = tf.nn.softplus(tau) + self.tau_epsilon

        return phi, tau, t_interp

    def call(self, inputs, mask=None, training=None):
        '''
        Forward pass for LiquidAttention.

        Args:
          inputs: can be
            - single tensor (self-attention)
            - tuple of 2 tensors ((q, k, v), t)
            - tuple of 3 tensors (q, k, v)
            - tuple of 4 tensors (q, k, v, t)
          mask: optional attention mask
          training: flag for dropout
        Returns:
          Attention output (and optionally weights)
        '''
        t = tf.constant(1.0, dtype=tf.float32)
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, t = inputs
            q_in = k_in = v_in = x
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            q_in, k_in, v_in = inputs
        elif isinstance(inputs, (list, tuple)) and len(inputs) == 4:
            q_in, k_in, v_in, t = inputs
            t = tf.cast(t, tf.float32)
        else:
            q_in = k_in = v_in = inputs

        # Linear projections
        q = self.q_dense(q_in)
        k = self.k_dense(k_in)
        v = self.v_dense(v_in)

        # Split heads
        qh = self.split_heads(q)
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        # Compute phi, tau, time
        phi, tau, time = self.compute_phi_tau(qh, kh, t)
        dt = tf.cast(self.delta_t, phi.dtype)

        # Solve dynamics based on chosen mode
        if self.mode == 'steady':
            attn_logits = phi / tau

        elif self.mode == 'exact':
            attn_logits = (phi / tau) * (1 - tf.exp(-tau * time))

        elif self.mode == 'euler':
            a = tf.zeros_like(phi)
            for _ in range(self.euler_steps):
                increment = dt * (-tau * a + phi)
                a = a + increment
            attn_logits = a

        # Apply mask
        if mask is not None:
            mask = tf.cast(mask, attn_logits.dtype)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)  # [batch, 1, 1, seq_len]
            very_neg = tf.constant(-1e9, dtype=attn_logits.dtype)
            attn_logits = attn_logits + (1.0 - mask) * very_neg

        # Softmax normalization
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Integrated output
        output = tf.matmul(attn_weights, vh)

        # Combine heads and final projection
        combined = self.combine_heads(output)
        out = self.activation(self.out_dense(combined))

        result = out if self.return_sequences else out[:, -1, :]
        if self.return_attention:
            return result, attn_weights
        return result


    def get_config(self):
        '''Return layer configuration for serialization.'''
        cfg = super().get_config()
        cfg.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'phi_hidden': self.phi_hidden,
            'tau_epsilon': self.tau_epsilon,
            'mode': self.mode,
            'euler_steps': self.euler_steps,
            'delta_t': self.delta_t,
            'dropout': self.dropout,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'return_attention': self.return_attention,
            'return_sequences': self.return_sequences
        })
        return cfg

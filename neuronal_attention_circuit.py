import math
import numpy as np
import tensorflow as tf
from ncps.wirings import AutoNCP #NCP wiring



@tf.keras.utils.register_keras_serializable(package="neuronal_attention_circuit",name="NAC")
class NAC(tf.keras.layers.Layer):
    '''
    Neuronal Attention Circuit (NAC)

    The NAC extends the Liquid Attention Network by sparsifying the pairwise concatenation 
    to the top-k elements. It projects queries, keys, and values through NCP-based sensory 
    neurons, then computes phi and tau through inter → command → motor pathways. The outputs 
    are integrated over time to enable dynamic attention. This design remains consistent with 
    the original LAN formulation while emphasizing computational clarity and efficiency.

    Uses AutoNCP from ncps.wiring (https://github.com/mlech26l/ncps/blob/master/ncps/wirings/wirings.py)
    as internal mechanism for constructing NCP-based neurons for computation.

    '''

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
        '''
        Initialize the NAC layer.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            topk: Number of top key candidates per query
            mode: computation mode ('steady', 'euler', or 'exact')
            euler_steps: Number of Euler integration steps (if mode='euler')
            sparsity: Sparsity factor for AutoNCP wiring (0.0–0.9)
            dt: Time step used for integration
            delta_t: Time increment for Euler integration
            activation: Optional activation function for the output
            tau_epsilon: Small constant added to tau for stability
            dropout: Dropout rate for attention weights
            use_bias: Whether to include bias terms in Dense layers
            return_attention: If True, call() also returns attention weights
            return_sequences: If False, only the final time step is returned
        '''
        super().__init__(**kwargs)

        # Basic checks and parameter setup
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert 0.0 <= sparsity <= 0.9, "sparsity must be in [0.0, 0.9]"
        assert mode in ('steady', 'euler', 'exact'), "mode must be 'steady', 'euler' or 'exact'"

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

        # Projections for query, key, and value
        self.q_proj = self._make_sensory_projections("q_proj")
        self.k_proj = self._make_sensory_projections("k_proj")
        self.v_proj = self._make_sensory_projections("v_proj")

        # Small time-based MLPs
        self.time_a = tf.keras.layers.Dense(1, use_bias=True, name="time_a")
        self.time_b = tf.keras.layers.Dense(1, use_bias=True, name="time_b")

        # NCP output that compute phi and tau
        self.out_ncp = self._make_inter_to_motor_projections("out")

        # Dropout applied to attention weights
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        #attention out projection
        self.out_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="out_proj")


    # ---------------- utility methods for multi-head ops ----------------
    def split_heads(self, x):
        '''
        Reshape [B, T, d_model] into [B, H, T, depth] for multi-head attention.
        '''
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        x = tf.reshape(x, (B, T, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        '''
        Reshape [B, H, T, depth] back into [B, T, d_model].
        '''
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        return tf.reshape(x, (B, T, self.d_model))

    # ---------------- sparse top-k attention selection ----------------
    def sparse_topk_pairwise(self, q, k, K=None):
        '''
        Compute pairwise dot products between q and k, and select the top-K keys per query.

        Args:
            q: Query tensor of shape [B, H, Tq, D]
            k: Key tensor of shape [B, H, Tk, D]
            K: Optional override for top-k value

        Returns:
            topk_pairs: Concatenated [q, selected_k] pairs, shape [B, H, Tq, K, 2D]
            topk_idx: Indices of selected keys, shape [B, H, Tq, K]
        '''
        if K is None:
            K = self.topk

        scores = tf.einsum("bhqd,bhkd->bhqk", q, k)
        Tk = tf.shape(k)[2]
        K_eff = tf.minimum(K, Tk)

        _, topk_idx = tf.math.top_k(scores, k=K_eff)
        k_gathered = tf.gather(k, topk_idx, batch_dims=2, axis=2)
        q_tiled = tf.tile(tf.expand_dims(q, axis=3), [1, 1, 1, K_eff, 1])

        topk_pairs = tf.concat([q_tiled, k_gathered], axis=-1)
        return topk_pairs, topk_idx

    # ---------------- projection helpers ----------------
    def _make_sensory_projections(self, name: str):
        '''
        Create an RNN for sensory projections (used for q, k, v).
        '''
        sensory_units = math.ceil((self.d_model - 0.5) / 0.6)
        wiring = AutoNCP(sensory_units, 0, sparsity_level=self.sparsity)
        cell = NCPCell(
            wiring,
            activation="linear",
            input_group="sensory",
            output_group="sensory",
            disabled_groups=["inter", "command", "motor"]
        )
        return tf.keras.layers.RNN(cell, return_sequences=True, name=name)

    def _make_inter_to_motor_projections(self, name: str):
        '''
        Create an RNN mapping inter neurons to motor neurons (used for phi and tau).
        '''
        units = self.d_model + math.floor(self.d_model / 0.6)
        wiring = AutoNCP(units, 1, sparsity_level=self.sparsity)
        cell = NCPCell(
            wiring,
            activation="linear",
            input_group="inter",
            output_group="motor",
            disabled_groups=["sensory"]
        )
        return tf.keras.layers.RNN(cell, return_sequences=True, name=name)

    # ---------------- phi, tau, and time interpolation ----------------
    def compute_phi_tau(self, q, k, t):
        '''
        Compute phi (gating), tau (time constant), and time interpolation factors for top-k pairs.

        Args:
            q: Query tensor [B, H, Tq, D]
            k: Key tensor [B, H, Tk, D]
            t: Time scalar or tensor

        Returns:
            phi: [B, H, Tq, K]
            tau: [B, H, Tq, K]
            t_interp: [B, H, Tq, K]
            topk_idx: Key indices used in selection
        '''
        B = tf.shape(q)[0]
        H = tf.shape(q)[1]
        Tq = tf.shape(q)[2]

        pair, topk_idx = self.sparse_topk_pairwise(q, k, K=self.topk)
        K_eff = tf.shape(pair)[3]
        D2 = tf.shape(pair)[-1]

        flat = tf.reshape(pair, [-1, D2])
        flat_3d = tf.expand_dims(flat, axis=1)

        out_raw = self.out_ncp(flat_3d)
        phi = tf.reshape(tf.nn.sigmoid(out_raw), [B, H, Tq, K_eff])
        tau = tf.reshape(tf.nn.softplus(out_raw) + self.tau_epsilon, [B, H, Tq, K_eff])

        t_a = self.time_a(pair)
        t_b = self.time_b(pair)

        t_cast = tf.cast(t, pair.dtype)
        t_expanded = tf.reshape(t_cast, [-1 if t_cast.shape.rank else 1, 1, 1, 1, 1])
        t_interp = tf.nn.sigmoid(t_a * t_expanded + t_b)[..., 0]

        return phi, tau, t_interp, topk_idx

    # ---------------- forward pass ----------------
    def call(self, inputs, mask=None, training=None):
        '''
        Forward computation of the NAC layer.

        Input formats:
          - x → q, k, v = x; t = 1.0
          - (x, t)
          - (q, k, v)
          - (q, k, v, t)

        Returns:
            Output tensor or (output, attention_weights) if return_attention=True.
        '''
        # Parse flexible input formats
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                x, t = inputs
                q_in = k_in = v_in = x
            elif len(inputs) == 3:
                q_in, k_in, v_in = inputs
                t = tf.constant(1.0, dtype=tf.float32)
            elif len(inputs) == 4:
                q_in, k_in, v_in, t = inputs
            else:
                raise ValueError("Unsupported input tuple length")
        else:
            q_in = k_in = v_in = inputs
            t = tf.constant(1.0, dtype=tf.float32)

        # Project q, k, v through sensory neurons
        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)

        # Split into multiple heads
        qh = self.split_heads(q)
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        # Compute phi, tau, and time interpolation
        phi, tau, t_interp, topk_idx = self.compute_phi_tau(qh, kh, t)


        # Solve dynamics based on chosen mode
        if self.mode == 'steady':
            attn_logits = phi / tau

        elif self.mode == 'exact':
            attn_logits = (phi / tau) * (1 - tf.exp(-tau * t_interp))

        elif self.mode == 'euler':
            a = tf.zeros_like(phi)
            for _ in range(self.euler_steps):
                increment = self.delta_t * (-tau * a + phi)
                a = a + increment
            attn_logits = a

        # Apply mask if provided
        if mask is not None:
            mask = tf.cast(mask, attn_logits.dtype)
            mask_exp = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            mask_gathered = tf.gather(mask_exp, topk_idx, batch_dims=2, axis=2)
            attn_logits = attn_logits * mask_gathered

        # Normalize and apply dropout
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Gather values corresponding to selected keys
        vh_topk = tf.gather(vh, topk_idx, batch_dims=2, axis=2)

        # integration
        weighted = attn_weights[..., tf.newaxis] * vh_topk
        out_per_head = tf.reduce_sum(weighted, axis=3) * self.dt

        # Combine heads and project output
        combined = self.combine_heads(out_per_head)
        out = self.out_dense(combined)
        if self.activation is not None:
            out = self.activation(out)

        # Optionally return only the last time step
        if not self.return_sequences:
            out = out[:, -1, :]

        if self.return_attention:
            return out, attn_weights
        return out

    # ---------------- configuration serialization ----------------
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


@tf.keras.utils.register_keras_serializable(package="neuronal_attention_circuit",name="NCPCell")
class NCPCell(tf.keras.layers.Layer):
    '''
    Neuronal Circuit Policy (NCP) recurrent cell.

    Implements a recurrent neural network cell based on worm-brain inspired
    connectivity patterns. Supports selective activation or disabling of neuron
    groups and uses sparse connectivity defined by the wiring object.
    '''

    def __init__(
        self,
        wiring,
        activation="linear",
        input_group="sensory",
        output_group="motor",
        disabled_groups=None,
        **kwargs
    ):
        '''
        Initialize the NCP cell.

        Args:
            wiring: Wiring object defining adjacency matrices and neuron groups.
            activation: Activation function used for neuron updates.
            input_group: Name of the input neuron group.
            output_group: Name of the output neuron group.
            disabled_groups: List of neuron groups to deactivate
                (e.g., ['inter', 'command']).
        '''
        super().__init__(**kwargs)
        self.wiring = wiring
        self.activation = tf.keras.activations.get(activation)
        self.disabled_groups = disabled_groups if disabled_groups is not None else []
        self.output_group = output_group
        self.input_group = input_group

    # -------------------- properties --------------------
    @property
    def state_size(self):
        return self.wiring.units

    @property
    def sensory_size(self):
        return self.wiring.input_dim

    @property
    def motor_size(self):
        return self.wiring.output_dim

    @property
    def output_size(self):
        '''Number of output neurons, determined by the output group.'''
        return len(getattr(self, 'output_indices', [])) or self.motor_size

    # -------------------- build --------------------
    def build(self, input_shape):
        '''
        Create trainable parameters and neuron group indices based on the wiring.

        This method constructs all weights, masks, and indexing structures
        necessary for running the recurrent computation efficiently.
        '''
        if isinstance(input_shape[0], (tuple, tf.Tensor)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        # Prepare wiring for the specified input size
        self.wiring.build(input_dim)

        # Identify neuron groups
        sensory_adj = np.abs(self.wiring.sensory_adjacency_matrix)
        self.sensory_indices = np.where(np.sum(sensory_adj, axis=0) > 0)[0]
        self.motor_indices = np.arange(self.wiring.output_dim)

        # Derive inter and command neuron indices
        command_full = np.setdiff1d(
            np.arange(self.wiring.units),
            np.union1d(self.sensory_indices, self.motor_indices)
        )
        if len(command_full) > 0:
            cmd_adj = np.abs(self.wiring.adjacency_matrix)[command_full[:, None], command_full]
            incoming = np.sum(cmd_adj, axis=0) > 0
            self.inter_indices = command_full[~incoming]
            self.command_indices = command_full[incoming]
        else:
            self.inter_indices = np.array([], dtype=int)
            self.command_indices = np.array([], dtype=int)

        # Disable neuron groups as requested
        disabled_indices = np.array([], dtype=int)
        for group in self.disabled_groups:
            if group == 'sensory':
                disabled_indices = np.union1d(disabled_indices, self.sensory_indices)
            elif group == 'inter':
                disabled_indices = np.union1d(disabled_indices, self.inter_indices)
            elif group == 'command':
                disabled_indices = np.union1d(disabled_indices, self.command_indices)
            elif group == 'motor':
                disabled_indices = np.union1d(disabled_indices, self.motor_indices)
            else:
                raise ValueError(f"Unknown group to disable: {group}")

        # Create mask to deactivate specified neurons
        active_mask_value = np.ones((self.state_size,), dtype="float32")
        active_mask_value[disabled_indices] = 0.0
        self.active_mask = self.add_weight(
            name="active_mask",
            shape=(self.state_size,),
            dtype="float32",
            initializer=tf.keras.initializers.Constant(active_mask_value),
            trainable=False
        )

        # Map group names to their neuron indices
        group_map = {
            'sensory': self.sensory_indices,
            'inter': self.inter_indices,
            'command': self.command_indices,
            'motor': self.motor_indices,
            'all': np.arange(self.state_size)
        }

        if self.input_group not in group_map:
            raise ValueError(f"Unknown input_group: {self.input_group}")
        if self.output_group not in group_map:
            raise ValueError(f"Unknown output_group: {self.output_group}")

        self.input_indices = group_map[self.input_group]
        self.output_indices = group_map[self.output_group]

        # ---------------- parameters ----------------
        self._params = {}

        # Input and recurrent kernels
        self._params["input_kernel"] = self.add_weight(
            name="input_kernel",
            shape=(self.sensory_size, self.state_size),
            dtype="float32",
            initializer=tf.keras.initializers.GlorotUniform(),
        )
        self._params["recurrent_kernel"] = self.add_weight(
            name="recurrent_kernel",
            shape=(self.state_size, self.state_size),
            dtype="float32",
            initializer=tf.keras.initializers.Orthogonal(),
        )
        self._params["bias"] = self.add_weight(
            name="bias",
            shape=(self.state_size,),
            dtype="float32",
            initializer=tf.keras.initializers.Zeros(),
        )

        # Fixed sparse connectivity masks
        sparsity_mask_value = np.abs(self.wiring.adjacency_matrix)
        self._params["sparsity_mask"] = self.add_weight(
            name="sparsity_mask",
            shape=sparsity_mask_value.shape,
            dtype="float32",
            initializer=tf.keras.initializers.Constant(sparsity_mask_value),
            trainable=False
        )

        sensory_mask = np.zeros((self.sensory_size, self.state_size), dtype="float32")
        sensory_mask[:, self.input_indices] = 1.0
        self._params["sensory_sparsity_mask"] = self.add_weight(
            name="sensory_sparsity_mask",
            shape=sensory_mask.shape,
            dtype="float32",
            initializer=tf.keras.initializers.Constant(sensory_mask),
            trainable=False
        )

        # Input and output affine transformations
        self._params["input_w"] = self.add_weight(
            name="input_w",
            shape=(self.sensory_size,),
            dtype="float32",
            initializer=tf.keras.initializers.Constant(1.0),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            shape=(self.sensory_size,),
            dtype="float32",
            initializer=tf.keras.initializers.Zeros(),
        )
        self._params["output_w"] = self.add_weight(
            name="output_w",
            shape=(len(self.output_indices),),
            dtype="float32",
            initializer=tf.keras.initializers.Constant(1.0),
        )
        self._params["output_b"] = self.add_weight(
            name="output_b",
            shape=(len(self.output_indices),),
            dtype="float32",
            initializer=tf.keras.initializers.Zeros(),
        )

        self.built = True

    # -------------------- helpers --------------------
    def _map_inputs(self, inputs):
        '''Apply an affine transform to sensory inputs.'''
        return inputs * self._params["input_w"] + self._params["input_b"]

    def _map_outputs(self, state):
        '''Select outputs from the neuron state and apply scaling.'''
        output = tf.gather(state, self.output_indices, axis=1)
        return output * self._params["output_w"] + self._params["output_b"]

    # -------------------- forward pass --------------------
    def call(self, inputs, states, training=False):
        '''
        Perform one recurrent step.

        Args:
            inputs: Current sensory input tensor with shape [B, sensory_size].
            states: List containing the previous neuron state [B, state_size].
        '''
        inputs = self._map_inputs(inputs)
        state = states[0]

        # Compute recurrent and sensory contributions using sparse connectivity
        recurrent = tf.matmul(
            state, self._params["recurrent_kernel"] * self._params["sparsity_mask"]
        )
        sensory = tf.matmul(
            inputs, self._params["input_kernel"] * self._params["sensory_sparsity_mask"]
        )

        # Update neuron states, apply activation, and enforce active mask
        next_state = self.activation(recurrent + sensory + self._params["bias"])
        next_state = next_state * self.active_mask

        outputs = self._map_outputs(next_state)
        return outputs, [next_state]

    # -------------------- serialization --------------------
    def get_config(self):
        '''
        Return configuration for saving and loading the layer.
        '''
        config = super().get_config()
        config.update({
            "wiring": self.wiring.get_config(),
            "activation": tf.keras.activations.serialize(self.activation),
            "disabled_groups": self.disabled_groups,
            "output_group": self.output_group,
            "input_group": self.input_group,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

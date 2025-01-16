import tensorflow as tf
from tensorflow.keras import layers


class LearnedL2NormPooling(layers.Layer):
    def __init__(self, kernel_size=2, strides=2, epsilon=1e-6):
        super(LearnedL2NormPooling, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.epsilon = epsilon

    def build(self, input_shape):
        self.alpha = self.add_weight(
            'alpha',
            shape=[1],
            initializer=tf.keras.initializers.Constant(1.0),  # Start with 1
            constraint=tf.keras.constraints.NonNeg(),  # Ensure alpha stays positive
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # Add debugging
        tf.debugging.check_numerics(inputs, 'Input to L2Pool')

        # Clip inputs to prevent extreme values
        x = tf.clip_by_value(inputs, -1e4, 1e4)

        # Square the inputs
        x = tf.square(x)

        # Check after square
        tf.debugging.check_numerics(x, 'After square in L2Pool')

        # Apply average pooling to squared values
        x = tf.nn.avg_pool1d(
            x,
            ksize=self.kernel_size,
            strides=self.strides,
            padding='VALID'
        )

        # Check after pooling
        tf.debugging.check_numerics(x, 'After pooling in L2Pool')

        # Apply learned scale and square root with epsilon
        x = x * tf.maximum(self.alpha, self.epsilon)
        x = tf.sqrt(tf.maximum(x, self.epsilon))

        return x


class TransformerBlock(layers.Layer):
    def __init__(self, hidden_size, num_heads, ff_dim, dropout, epsilon, is_training):
        super(TransformerBlock, self).__init__()
        self.is_training = is_training
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(hidden_size)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=epsilon)
        self.layernorm2 = layers.LayerNormalization(epsilon=epsilon)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, mask=None):
        # Multi-head attention with residual connection and layer norm
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=self.is_training)
        out1 = self.layernorm1(inputs + attention_output)

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=self.is_training)
        return self.layernorm2(out1 + ffn_output)


class PowerAwareAttention(layers.MultiHeadAttention):
    """
    PowerAwareAttention leverages the attention mechanism specifically tailored for Non-Intrusive Load Monitoring (NILM)
    tasks. This class introduces a power-awareness mechanism that integrates power consumption thresholds and power
    state transitions to enhance the disaggregation process.

    Key Features:\n
    1. **Base Attention Mechanism**: Extends the standard multi-head attention to capture temporal dependencies in power data.
    2. **Power Threshold Masking**: Dynamically adjusts attention scores based on whether power values exceed a specified threshold.
    3. **Transition Detection**: Highlights sudden changes in power states (ON to OFF or OFF to ON) and uses them to adjust attention scores.
    4. **Event-Based Attention**: Encodes and computes similarity scores for power events, reinforcing the attention mechanism where
       significant power transitions or events occur.
    5. **Combined Attention Scores**: The final attention scores are a combination of base attention, power-based masking,
       transition-based weighting, and event-based similarity, which makes the mechanism more robust for NILM tasks.

    Parameters:
    - num_heads: Number of attention heads.
    - key_dim: Size of each attention head for queries and keys.
    - power_threshold: Power value threshold for creating the power-based mask.
    - value_dim: Size of each attention head for values (default is key_dim).
    - dropout: Dropout rate for attention weights.
    - use_bias: Whether to use bias in attention layers.
    - output_shape: Desired output shape of the attention layer.
    - attention_axes: Axes over which the attention is applied.

    This class improves NILM disaggregation by focusing attention on key power events and transitions, providing better appliance-level
    separation and reducing noise in the aggregated power data.
    """

    def __init__(
            self,
            num_heads,
            key_dim,
            power_threshold,
            value_dim=None,
            dropout=0.0,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            kernel_regularizer=None,
            **kwargs
    ):
        super(PowerAwareAttention, self).__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=output_shape,
            attention_axes=attention_axes,
            kernel_regularizer=kernel_regularizer,
            **kwargs
        )

        self.kernel_regularizer = kernel_regularizer
        self.power_threshold = power_threshold

        # Ensure value_dim is set
        self._value_dim = value_dim if value_dim is not None else key_dim

        # Call _build_from_signature to set up necessary attributes
        self._build_from_signature(query=tf.TensorSpec(shape=[None, None, key_dim]),
                                   value=tf.TensorSpec(shape=[None, None, self._value_dim]),
                                   key=tf.TensorSpec(shape=[None, None, key_dim]))

        # Add a dense layer to project the attention output back to the original dimension
        self.output_dense = layers.Dense(key_dim * num_heads)

    def build(self, input_shape):
        super(PowerAwareAttention, self).build(input_shape)

        # Build event detection components
        self.event_encoder = layers.Dense(self._key_dim * self._num_heads)

        # Temperature parameter for attention scaling
        self.temperature = self.add_weight(
            'temperature',
            shape=[1],
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        # Compute base attention scores using parent class method
        query_shape = tf.shape(query)
        batch_size, seq_len = query_shape[0], query_shape[1]

        # Reshape query, key, value
        query = tf.reshape(query, [batch_size, seq_len, self._num_heads, self._key_dim])
        key = tf.reshape(key, [batch_size, seq_len, self._num_heads, self._key_dim])
        value = tf.reshape(value, [batch_size, seq_len, self._num_heads, self._value_dim])

        # Transpose to [batch, num_heads, seq_len, key/value_dim]
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        # Compute base attention scores
        base_scores = tf.matmul(query, key, transpose_b=True)
        base_scores = base_scores / tf.math.sqrt(tf.cast(self._key_dim, dtype=base_scores.dtype))

        # Extract power values (assuming first feature is power)
        power_signal = value[:, :, :, 0]  # [batch, num_heads, seq_len]

        # Create power-based attention mask
        power_mask = tf.cast(power_signal > self.power_threshold, tf.float32)
        power_mask = tf.expand_dims(power_mask, axis=-2)  # [batch, num_heads, 1, seq_len]

        # Compute power transitions (ON->OFF and OFF->ON)
        power_transitions = tf.abs(power_signal[:, :, 1:] - power_signal[:, :, :-1])
        power_transitions = tf.pad(power_transitions, [[0, 0], [0, 0], [0, 1]])
        transition_mask = tf.cast(power_transitions > self.power_threshold, tf.float32)
        transition_mask = tf.expand_dims(transition_mask, axis=-2)  # [batch, num_heads, 1, seq_len]

        # Compute event-based attention
        event_input = tf.reshape(value, [batch_size, seq_len, self._num_heads * self._value_dim])
        event_features = self.event_encoder(event_input)
        event_features = tf.reshape(event_features, [batch_size, seq_len, self._num_heads, self._key_dim])
        event_features = tf.transpose(event_features, [0, 2, 1, 3])  # [batch, num_heads, seq_len, key_dim]

        # Compute event similarity scores
        event_scores = tf.matmul(event_features, event_features, transpose_b=True)
        event_scores = event_scores / tf.sqrt(tf.cast(self._key_dim, tf.float32))

        # Combine attention mechanisms
        combined_scores = (base_scores +
                           power_mask * 1.0 +
                           transition_mask * 2.0 +
                           event_scores * 0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = tf.expand_dims(tf.cast(attention_mask, tf.float32), axis=1)
            mask = tf.tile(mask, [1, self._num_heads, 1, 1])
            combined_scores += (1.0 - mask) * -1e9

        # Temperature-scaled softmax
        attention_weights = tf.nn.softmax(combined_scores / self.temperature, axis=-1)

        if training and self._dropout > 0:
            attention_weights = tf.nn.dropout(attention_weights, self._dropout)

        # Apply attention weights to values
        attention_output = tf.matmul(attention_weights, value)  # [batch, num_heads, seq_len, value_dim]

        # Transpose and reshape the output
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [batch_size, seq_len, self._num_heads * self._value_dim])

        # Project back to the original dimension
        attention_output = self.output_dense(attention_output)

        return attention_output, attention_weights

    def call(self, query, value, key=None, attention_mask=None, training=None, return_attention_scores=False):
        if key is None:
            key = value

        attention_output, attention_weights = self._compute_attention(
            query, key, value,
            attention_mask=attention_mask,
            training=training
        )

        if return_attention_scores:
            return attention_output, attention_weights
        return attention_output

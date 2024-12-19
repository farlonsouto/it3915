import tensorflow as tf
from tensorflow.keras import layers, Model


class LearnedL2NormPooling(layers.Layer):
    def __init__(self, pool_size=2, epsilon=1e-6, **kwargs):
        super(LearnedL2NormPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            name='l2_norm_weight',
            shape=(1, 1, input_shape[-1]),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        squared_inputs = tf.square(inputs)
        pooled = tf.nn.avg_pool1d(
            squared_inputs,
            ksize=self.pool_size,
            strides=self.pool_size,
            padding='SAME'
        )
        weighted_pooled = pooled * self.weight
        return tf.sqrt(weighted_pooled + self.epsilon)


class PFFN(layers.Layer):
    def __init__(self, hidden_size, ff_dim, **kwargs):
        super(PFFN, self).__init__(**kwargs)
        self.dense1 = layers.Dense(ff_dim, activation='gelu')
        self.dense2 = layers.Dense(hidden_size)

    def call(self, x):
        return self.dense2(self.dense1(x))


class TransformerBlock(layers.Layer):
    def __init__(self, hidden_size, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.ffn = PFFN(hidden_size, ff_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class BERT4NILM(Model):
    def __init__(self, appliance_max_power, window_size, wandb_config, hidden_size=256, num_layers=2, num_heads=2,
                 ff_dim=1024, dropout_rate=0.1):
        super(BERT4NILM, self).__init__()

        self.wandb_config = wandb_config
        # Use a small negative value instead of extreme -1000
        self.MASK = 0.0  # Changed to 0 since we'll use attention masking

        self.appliance_max_power = appliance_max_power
        self.window_size = window_size
        self.hidden_size = hidden_size

        # Input embedding layers
        self.embedding = layers.Dense(hidden_size)
        self.pos_embedding = self.add_weight("pos_embedding", shape=(1, window_size, hidden_size))

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(hidden_size, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]

        # Output layers
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_dense = layers.Dense(1)

    def create_padding_mask(self, mask):
        """Create attention padding mask"""
        return tf.cast(tf.math.not_equal(mask, self.MASK), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def call(self, inputs, training=True, mask=None):
        # Input shape: [batch_size, seq_len, features]
        x = self.embedding(inputs)

        # Add positional encoding
        x = x + self.pos_embedding

        # Apply transformer blocks with attention masking
        attention_mask = None
        if mask is not None:
            attention_mask = self.create_padding_mask(mask)

        for transformer in self.transformer_blocks:
            x = transformer(x, training=training, mask=attention_mask)

        x = self.layer_norm(x)
        outputs = self.output_dense(x)

        # Scale back to original power values
        return tf.clip_by_value(outputs * self.appliance_max_power, 0, self.appliance_max_power)

    def train_step(self, data):
        x, y = data

        # Create masking pattern
        # Instead of random masking, use structured masking
        mask = self.create_structured_mask(x)
        x_masked = tf.where(mask, self.MASK, x)

        with tf.GradientTape() as tape:
            y_pred = self(x_masked, training=True, mask=mask)
            loss = self.compute_loss(y, y_pred, mask)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def create_structured_mask(self, x):
        """Create structured masking pattern for time series"""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Create spans of masked values instead of random points
        mask_len = tf.random.uniform([], minval=1, maxval=5, dtype=tf.int32)
        num_masks = tf.cast(seq_len * self.wandb_config.masking_portion / tf.cast(mask_len, tf.float32), tf.int32)

        mask = tf.zeros((batch_size, seq_len))

        for i in range(batch_size):
            # Generate random starting points for masks
            start_points = tf.random.uniform([num_masks], 0, seq_len - mask_len, dtype=tf.int32)

            for start in start_points:
                end = tf.minimum(start + mask_len, seq_len)
                indices = tf.range(start, end)
                updates = tf.ones([end - start])
                mask = tf.tensor_scatter_nd_update(
                    mask,
                    tf.stack([tf.ones_like(indices) * i, indices], axis=1),
                    updates
                )

        return tf.cast(mask, tf.bool)

    def compute_loss(self, y_true, y_pred, mask):
        """Compute masked loss"""
        # Only compute loss on masked positions
        mask_float = tf.cast(mask, tf.float32)

        # MSE loss on masked positions
        mse = tf.reduce_mean(
            mask_float * tf.square(y_true - y_pred)
        )

        # Add regularization term for temporal consistency
        temp_consistency = tf.reduce_mean(
            tf.square(y_pred[:, 1:] - y_pred[:, :-1])
        )

        return mse + 0.1 * temp_consistency

import tensorflow as tf
from tensorflow.keras import layers, Model


class PositionWiseFFN(layers.Layer):
    def __init__(self, hidden_size, ff_dim, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = layers.Dense(ff_dim, activation='gelu')
        self.dense2 = layers.Dense(hidden_size)

    def call(self, x, **kwargs):
        return self.dense2(self.dense1(x))


class TransformerBlock(layers.Layer):
    def __init__(self, hidden_size, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.ffn = PositionWiseFFN(hidden_size, ff_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=True, mask=None):
        # Pass the attention mask to MultiHeadAttention
        attn_output = self.att(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask
        )
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
        return tf.cast(mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]

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
        return tf.clip_by_value(outputs * self.appliance_max_power, 1.0, self.appliance_max_power)

    def train_step(self, data):
        x, y = data

        # Create masking pattern
        mask = self.create_structured_mask(x)  # Shape: [batch_size, seq_len]

        # Expand mask dimensions to match input shape
        mask_expanded = tf.expand_dims(mask, axis=-1)  # Shape: [batch_size, seq_len, 1]

        # Apply masking
        x_masked = tf.where(mask_expanded, tf.zeros_like(x), x)

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

        # Convert everything to float32 for the calculation, then back to int32
        seq_len_float = tf.cast(seq_len, tf.float32)
        mask_len_float = tf.cast(mask_len, tf.float32)

        # Calculate number of masks needed to achieve desired masking portion
        num_masks = tf.cast(
            seq_len_float * self.wandb_config.masking_portion / mask_len_float,
            tf.int32
        )

        # Initialize mask tensor
        mask = tf.zeros((batch_size, seq_len))

        # Generate random starting points for all batches at once
        start_points = tf.random.uniform(
            [batch_size, num_masks],
            minval=0,
            maxval=seq_len - mask_len,
            dtype=tf.int32
        )

        # Create indices for batch dimension
        batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
        batch_indices = tf.tile(batch_indices, [1, num_masks, mask_len])

        # Create mask_len sequences for each start point
        offsets = tf.range(mask_len)[tf.newaxis, tf.newaxis, :]
        seq_indices = start_points[..., tf.newaxis] + offsets

        # Combine indices
        indices = tf.stack([
            tf.reshape(batch_indices, [-1]),
            tf.reshape(seq_indices, [-1])
        ], axis=1)

        # Create updates
        updates = tf.ones([batch_size * num_masks * mask_len])

        # Update mask using scatter
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        return tf.cast(mask, tf.bool)

    def compute_loss(self, y_true, y_pred, mask):
        """Compute masked loss"""
        # Convert boolean mask to float for loss computation and expand dimensions
        mask_float = tf.cast(mask, tf.float32)
        mask_float = tf.expand_dims(mask_float, axis=-1)

        # MSE loss on masked positions
        mse = tf.reduce_mean(
            mask_float * tf.square(y_true - y_pred)
        )

        # Add regularization term for temporal consistency
        temp_consistency = tf.reduce_mean(
            tf.square(y_pred[:, 1:] - y_pred[:, :-1])
        )

        return mse + 0.1 * temp_consistency

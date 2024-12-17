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
    def __init__(self, appliance_max_power, window_size, hidden_size=256, num_layers=2, num_heads=2, ff_dim=1024,
                 dropout_rate=0.1):
        super(BERT4NILM, self).__init__()

        self.MASK = -1.0

        self.appliance_max_power = appliance_max_power
        self.window_size = window_size
        self.hidden_size = hidden_size

        self.conv = layers.Conv1D(filters=hidden_size, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool = LearnedL2NormPooling(pool_size=2)
        self.pos_embedding = self.add_weight("pos_embedding", shape=(1, window_size // 2, hidden_size))

        self.transformer_blocks = [TransformerBlock(hidden_size, num_heads, ff_dim, dropout_rate) for _ in
                                   range(num_layers)]

        self.deconv = layers.Conv1DTranspose(filters=hidden_size, kernel_size=4, strides=2, padding='same',
                                             activation='relu')
        self.output_layer1 = layers.Dense(ff_dim, activation='tanh')
        self.output_layer2 = layers.Dense(1)

    def call(self, inputs, training=True, mask=None):

        # Ensure inputs have 3 dimensions: [batch_size, seq_len, num_features]
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)  # Add a channel dimension

        x = self.conv(inputs)
        x = self.pool(x)
        x += self.pos_embedding

        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)

        x = self.deconv(x)
        x = self.output_layer1(x)
        x = self.output_layer2(x)

        x = x * self.appliance_max_power
        return tf.clip_by_value(x, 1, self.appliance_max_power)

    def train_step(self, data):
        x, y = data

        # Ensure x and y have the correct shape
        # x = tf.squeeze(x, axis=-1)
        # y = tf.squeeze(y, axis=-1)

        # Apply masking
        mask = tf.random.uniform(shape=tf.shape(x)) < 0.25
        x_masked = tf.where(mask, self.MASK, x)

        with tf.GradientTape() as tape:
            y_pred = self(x_masked, training=True)
            loss = self.compute_loss(y, y_pred, mask)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and log metrics
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss

        return metrics

    def compute_loss(self, y_true, y_pred, mask):
        """
        Compute loss over masked positions only.

        Args:
            y_true: Ground truth tensor, shape [batch_size, seq_len, 1].
            y_pred: Predicted tensor, shape [batch_size, seq_len, 1].
            mask: Mask tensor, shape [batch_size, seq_len, 1].

        Returns:
            Total loss.
        """
        # Ensure all tensors are aligned
        mask = tf.squeeze(mask, axis=-1)  # [batch_size, seq_len]
        y_true = tf.squeeze(y_true, axis=-1)  # [batch_size, seq_len]
        y_pred = tf.squeeze(y_pred, axis=-1)  # [batch_size, seq_len]

        # Apply the mask to extract relevant values
        y_true_masked = tf.boolean_mask(y_true, mask)  # [num_masked_positions]
        y_pred_masked = tf.boolean_mask(y_pred, mask)  # [num_masked_positions]

        # Compute Mean Squared Error
        mse_loss = tf.reduce_mean(tf.square(y_pred_masked - y_true_masked))

        # KL Divergence
        kl_div = tf.reduce_mean(
            0.5 * (
                    tf.square(y_pred_masked - y_true_masked)
                    - 1
                    + tf.math.log(1e-8 + tf.exp(1 - tf.square(y_pred_masked - y_true_masked)))
            )
        )

        # Soft-margin loss
        status_true = tf.cast(y_true_masked > 0, tf.float32)
        status_pred = tf.cast(y_pred_masked > 0, tf.float32)
        soft_margin = tf.reduce_mean(tf.nn.softplus(-status_true * status_pred))

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(y_pred_masked - y_true_masked))

        # Combine losses
        total_loss = mse_loss + 0.1 * kl_div + soft_margin + 0.001 * l1_loss
        return total_loss

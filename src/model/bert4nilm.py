import tensorflow as tf
import tensorflow_probability as tfp
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
    def __init__(self, max_power, window_size, hidden_size=256, num_layers=2, num_heads=2, ff_dim=512,
                 dropout_rate=0.1):
        super(BERT4NILM, self).__init__()
        self.max_power = max_power
        self.window_size = window_size
        self.hidden_size = hidden_size

        self.conv = layers.Conv1D(filters=hidden_size, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool = LearnedL2NormPooling(pool_size=2)
        self.pos_embedding = self.add_weight("pos_embedding", shape=(1, window_size // 2, hidden_size))

        self.transformer_blocks = [TransformerBlock(hidden_size, num_heads, ff_dim, dropout_rate) for _ in
                                   range(num_layers)]

        self.deconv = layers.Conv1DTranspose(filters=hidden_size, kernel_size=4, strides=2, padding='same',
                                             activation='relu')
        self.output_layer1 = layers.Dense(128, activation='tanh')
        self.output_layer2 = layers.Dense(1)

    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.pool(x)
        x += self.pos_embedding

        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)

        x = self.deconv(x)
        x = self.output_layer1(x)
        x = self.output_layer2(x)

        return tf.clip_by_value(x * self.max_power, 0, self.max_power)

    def train_step(self, data):
        x, y = data

        # Apply masking
        mask = tf.random.uniform(shape=tf.shape(x)) < 0.25
        x_masked = tf.where(mask, -1.0, x)

        with tf.GradientTape() as tape:
            y_pred = self(x_masked, training=True)
            loss = self.compute_loss(y, y_pred, mask)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def compute_loss(self, y_true, y_pred, mask):
        # MSE
        mse = tf.reduce_mean(tf.square(y_pred - y_true))

        # KL Divergence
        y_true_dist = tfp.distributions.Normal(y_true, scale=1.0)
        y_pred_dist = tfp.distributions.Normal(y_pred, scale=1.0)
        kl_div = tf.reduce_mean(tfp.distributions.kl_divergence(y_true_dist, y_pred_dist))

        # Soft-margin loss
        status_true = tf.cast(y_true > 0, tf.float32)
        status_pred = tf.cast(y_pred > 0, tf.float32)
        soft_margin = tf.reduce_mean(tf.nn.softplus(-status_true * status_pred))

        # L1 loss (only for on-state or misclassified)
        l1_mask = tf.logical_or(status_true > 0, tf.not_equal(status_true, status_pred))
        l1 = tf.reduce_mean(tf.abs(y_pred - y_true) * tf.cast(l1_mask, tf.float32))

        # Combine losses
        return mse + 0.1 * kl_div + soft_margin + 0.001 * l1

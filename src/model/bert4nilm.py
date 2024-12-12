import tensorflow as tf
from keras import layers, Model


# tf.config.run_functions_eagerly(True)  # Forces eager execution in tf.function


class LearnedL2NormPooling(layers.Layer):
    def __init__(self, pool_size=2, epsilon=1e-6, **kwargs):
        super(LearnedL2NormPooling, self).__init__(**kwargs)
        self.weight = None
        self.pool_size = pool_size
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            name='l2_norm_weight',
            shape=(1, 1, input_shape[-1]),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs, **kwargs):
        if inputs.shape.ndims == 4:
            inputs = tf.squeeze(inputs, axis=-2)
        squared_inputs = tf.square(inputs)
        pooled = tf.nn.avg_pool1d(
            squared_inputs,
            ksize=self.pool_size,
            strides=self.pool_size,
            padding='SAME'
        )
        weighted_pooled = pooled * self.weight
        return tf.sqrt(weighted_pooled + self.epsilon)


class BERT4NILM(Model):
    def __init__(self, wandb_config):
        super(BERT4NILM, self).__init__()

        self.hyper_param = wandb_config
        self.latent_len = wandb_config.window_size // 2

        self.conv = layers.Conv1D(
            filters=self.hyper_param.hidden_size,
            kernel_size=self.hyper_param.conv_kernel_size,
            strides=self.hyper_param.conv_strides,
            activation=self.hyper_param.conv_activation,
            padding="same",
            kernel_initializer='truncated_normal',
            kernel_regularizer="l2"
        )

        self.pool = LearnedL2NormPooling()

        self.position = layers.Embedding(
            input_dim=self.latent_len,
            output_dim=self.hyper_param.hidden_size
        )

        self.dropout = layers.Dropout(self.hyper_param.dropout)

        self.transformer_blocks = [
            self.build_transformer_block() for _ in range(self.hyper_param.n_layers)
        ]

        self.deconv = layers.Conv1DTranspose(
            filters=self.hyper_param.hidden_size,
            kernel_size=self.hyper_param.deconv_kernel_size,
            strides=self.hyper_param.deconv_strides,
            activation=self.hyper_param.deconv_activation,
            padding="same",
            kernel_initializer='truncated_normal',
            kernel_regularizer="l2"
        )

        self.output_layer1 = layers.Dense(
            128,  # So to match the article's implementation of reference
            activation='linear',
            kernel_initializer='truncated_normal',
            kernel_regularizer="l2"
        )

        self.output_layer2 = layers.Dense(
            1,
            activation='linear',
            kernel_initializer='truncated_normal',
            kernel_regularizer="l2"
        )

    def build_transformer_block(self):
        inputs = layers.Input(shape=(None, self.hyper_param.hidden_size))
        x = layers.LayerNormalization(epsilon=self.hyper_param.layer_norm_epsilon)(inputs)
        attn_output = layers.MultiHeadAttention(
            kernel_initializer='truncated_normal',
            kernel_regularizer="l2",
            num_heads=self.hyper_param.num_heads,
            key_dim=self.hyper_param.hidden_size // self.hyper_param.num_heads
        )(x, x)
        attn_output = self.dropout(attn_output)
        out1 = layers.Add()([inputs, attn_output])
        x = layers.LayerNormalization(epsilon=self.hyper_param.layer_norm_epsilon)(out1)
        ff_output = layers.Dense(self.hyper_param.ff_dim, activation=self.hyper_param.dense_activation
                                 , kernel_initializer='truncated_normal',
                                 kernel_regularizer="l2")(x)
        ff_output = layers.Dense(self.hyper_param.hidden_size, kernel_initializer='truncated_normal',
                                 kernel_regularizer="l2")(ff_output)
        ff_output = self.dropout(ff_output)
        return Model(inputs, layers.Add()([out1, ff_output]))

    def call(self, inputs, training=None, mask=None):

        # Pass through conv and pooling layers
        x_token = self.pool(self.conv(inputs))

        sequence_length = tf.shape(x_token)[1]
        self.latent_len = sequence_length
        positions = tf.range(start=0, limit=self.latent_len, delta=1)
        positional_embedding = self.position(positions)
        embedding = x_token + positional_embedding

        if training:
            mask = tf.random.uniform(shape=tf.shape(embedding)[:2]) < self.hyper_param.masking_portion
            embedding = tf.where(mask[:, :, tf.newaxis], 0.0, embedding)

        x = self.dropout(embedding, training=training)

        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)

        x = self.deconv(x)
        x = self.output_layer1(x)
        x = tf.math.tanh(x)  # Tanh activation - This is critical: It is NOT the activation function of a layer
        pred_appl_power = self.output_layer2(x)

        # Multiply by max power and clamp
        # Apply upper limit of max_power to all values
        pred_appl_power = tf.clip_by_value(pred_appl_power * self.hyper_param.max_power, 1.0,
                                           self.hyper_param.max_power)

        return pred_appl_power

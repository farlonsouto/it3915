import tensorflow as tf
from tensorflow.keras import Model, layers


class BERT4NILM(Model):

    def __init__(self, config, is_training):
        super(BERT4NILM, self).__init__()
        self.config = config

        # Embedding module with convolutional layer
        self.conv1d = layers.Conv1D(
            filters=config['hidden_size'],
            kernel_size=config['conv_kernel_size'],
            strides=config['conv_strides'],
            padding='same',
            activation=config['conv_activation']
        )

        # L2 Norm Pooling (learned)
        self.l2_pool = LearnedL2NormPooling(
            kernel_size=2,
            strides=2
        )

        # Positional embedding
        self.pos_embedding = self.add_weight(
            "pos_embedding",
            shape=[config['window_size'] // 2, config['hidden_size']],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )

        # Transformer layers
        self.transformer_blocks = [
            TransformerBlock(
                config['hidden_size'],
                config['num_heads'],
                config['ff_dim'],
                config['dropout'],
                config['layer_norm_epsilon'],
                is_training
            ) for _ in range(config['num_layers'])
        ]

        # Output module
        self.deconv = layers.Conv1DTranspose(
            filters=config['hidden_size'],
            kernel_size=config['deconv_kernel_size'],
            strides=config['deconv_strides'],
            padding='same',
            activation=config['deconv_activation']
        )

        self.dense1 = layers.Dense(config['hidden_size'], activation='tanh')
        self.dense2 = layers.Dense(config['output_size'])

    def call(self, inputs, training=False, mask=None):
        # Embedding module
        x = self.conv1d(inputs)
        x = self.l2_pool(x)

        # Add positional embedding
        x = x + self.pos_embedding

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)

        # Output module
        x = self.deconv(x)
        x = self.dense1(x)
        x = self.dense2(x)

        # Ensure output is between 0 and 1 TODO: Is that correct?
        predictions = tf.clip_by_value(x, 1, self.config['appliance_max_power'])

        return predictions

    def train_step(self, data):
        x, y = data
        aggregated, mask = x
        with tf.GradientTape() as tape:
            predictions = self(aggregated, training=True)
            loss = self.compiled_loss(y, predictions, mask)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}


class LearnedL2NormPooling(layers.Layer):
    def __init__(self, kernel_size=2, strides=2):
        super(LearnedL2NormPooling, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.alpha = self.add_weight(
            'alpha',
            shape=[1],
            initializer='ones',
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # Square the inputs
        x = tf.square(inputs)

        # Apply average pooling to squared values
        x = tf.nn.avg_pool1d(
            x,
            ksize=self.kernel_size,
            strides=self.strides,
            padding='VALID'
        )

        # Apply learned scale and square root
        return tf.sqrt(x * self.alpha)


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

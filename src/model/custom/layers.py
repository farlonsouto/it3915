import tensorflow as tf
from tensorflow.keras import layers


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

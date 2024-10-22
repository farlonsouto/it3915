import tensorflow as tf
from tensorflow.keras import layers, Model


class BERT4NILM(Model):
    def __init__(self, wandb_config):

        super(BERT4NILM, self).__init__()
        self.args = wandb_config

        self.original_len = wandb_config.window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = wandb_config.dropout

        self.hidden = 256
        self.heads = 2
        self.n_layers = 2
        self.output_size = wandb_config.output_size

        # Convolution and pooling layers
        self.conv = layers.Conv1D(filters=self.hidden, kernel_size=5, padding='same', activation='relu')
        self.pool = layers.MaxPooling1D(pool_size=2)

        # Positional embedding (learnable)
        self.position = layers.Embedding(input_dim=self.latent_len, output_dim=self.hidden)

        # Dropout layer
        self.dropout = layers.Dropout(self.dropout_rate)

        # Transformer Encoder blocks using Keras' built-in MultiHeadAttention, LayerNormalization, and Dense layers
        self.transformer_blocks = [
            self.build_transformer_block() for _ in range(self.n_layers)
        ]

        # Deconvolution (Conv1DTranspose), and dense layers for final prediction
        self.deconv = layers.Conv1DTranspose(filters=self.hidden, kernel_size=4, strides=2, padding='same')
        self.linear1 = layers.Dense(128, activation='tanh')
        self.linear2 = layers.Dense(self.output_size)

    def build_transformer_block(self):
        """Construct a Transformer encoder block using Keras components."""
        inputs = layers.Input(shape=(None, self.hidden))

        # Multi-head attention layer
        attn_output = layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.hidden)(inputs, inputs)
        attn_output = layers.Dropout(self.dropout_rate)(attn_output)
        attn_output = layers.Add()([inputs, attn_output])  # Residual connection
        attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)  # Layer normalization

        # Feed-forward network
        ff_output = layers.Dense(self.hidden * 4, activation='gelu')(attn_output)
        ff_output = layers.Dense(self.hidden)(ff_output)
        ff_output = layers.Dropout(self.dropout_rate)(ff_output)
        ff_output = layers.Add()([attn_output, ff_output])  # Residual connection
        ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output)  # Layer normalization

        return Model(inputs, ff_output)

    def call(self, inputs, training=None, mask=None):
        # Expand dimensions and apply convolution and pooling
        x_token = self.pool(self.conv(tf.expand_dims(inputs, axis=-1)))

        # Add positional embeddings
        positions = tf.range(start=0, limit=tf.shape(x_token)[1], delta=1)
        embedding = x_token + self.position(positions)

        # Apply dropout, passing the training flag
        x = self.dropout(embedding, training=training)

        # Pass through each transformer encoder block
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)  # Pass training to ensure dropout is applied correctly

        # Deconvolution and final dense layers
        x = self.deconv(x)
        x = self.linear1(x)
        return self.linear2(x)

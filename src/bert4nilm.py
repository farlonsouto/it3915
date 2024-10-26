import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class LearnedL2NormPooling(layers.Layer):
    def __init__(self, pool_size=2, epsilon=1e-6, **kwargs):
        super(LearnedL2NormPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.epsilon = epsilon  # Small value to prevent NaN issues

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

        # Average pooling with stability epsilon
        pooled = tf.nn.avg_pool1d(
            squared_inputs,
            ksize=self.pool_size,
            strides=self.pool_size,
            padding='VALID'
        )

        weighted_pooled = pooled * self.weight

        # Adding epsilon to avoid NaNs in the sqrt operation
        return tf.sqrt(weighted_pooled + self.epsilon)



class BERT4NILM(Model):
    def __init__(self, wandb_config):
        super(BERT4NILM, self).__init__()
        self.args = wandb_config

        # Model configuration parameters
        self.original_len = wandb_config.window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = wandb_config.dropout
        self.hidden = wandb_config.head_size
        self.heads = wandb_config.num_heads
        self.n_layers = wandb_config.n_layers
        self.output_size = wandb_config.output_size
        self.masking_portion = wandb_config.masking_portion
        self.conv_kernel_size = wandb_config.conv_kernel_size
        self.deconv_kernel_size = wandb_config.deconv_kernel_size
        self.ff_dim = wandb_config.ff_dim
        self.layer_norm_epsilon = wandb_config.layer_norm_epsilon
        self.kernel_initializer = wandb_config.kernel_initializer
        self.bias_initializer = wandb_config.bias_initializer
        self.kernel_regularizer = self.get_regularizer(wandb_config.kernel_regularizer)
        self.bias_regularizer = self.get_regularizer(wandb_config.bias_regularizer)

        # Convolutional layers and learned pooling
        self.conv = layers.Conv1D(
            filters=self.hidden,
            kernel_size=self.conv_kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        self.pool = LearnedL2NormPooling()

        # Positional embeddings and dropout layer
        self.position = layers.Embedding(
            input_dim=self.latent_len,
            output_dim=self.hidden,
            embeddings_initializer=self.kernel_initializer
        )
        self.dropout = layers.Dropout(self.dropout_rate)

        # Transformer encoder blocks
        self.transformer_blocks = [
            self.build_transformer_block() for _ in range(self.n_layers)
        ]

        # Deconvolution and final dense layer
        self.deconv = layers.Conv1DTranspose(
            filters=self.hidden,
            kernel_size=self.deconv_kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        # This final dense layer outputs a value for each time step in the sequence
        self.output_layer = layers.Dense(
            1,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

    @staticmethod
    def get_regularizer(regularizer_config):
        if regularizer_config == 'l1':
            return regularizers.l1(l=0.01)
        elif regularizer_config == 'l2':
            return regularizers.l2(l=0.01)
        elif regularizer_config == 'l1_l2':
            return regularizers.l1_l2(l1=0.01, l2=0.01)
        else:
            return None

    def build_transformer_block(self):
        inputs = layers.Input(shape=(None, self.hidden))

        # Multi-head attention layer
        attn_output = layers.MultiHeadAttention(
            num_heads=self.heads,
            key_dim=self.hidden // self.heads
        )(inputs, inputs)
        attn_output = layers.Dropout(self.dropout_rate)(attn_output)
        attn_output = layers.Add()([inputs, attn_output])
        attn_output = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)(attn_output)

        # Feed-forward network
        ff_output = layers.Dense(
            self.ff_dim,
            activation='gelu',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )(attn_output)
        ff_output = layers.Dense(
            self.hidden,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )(ff_output)
        ff_output = layers.Dropout(self.dropout_rate)(ff_output)
        ff_output = layers.Add()([attn_output, ff_output])
        ff_output = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)(ff_output)

        return Model(inputs, ff_output)

    def call(self, inputs, training=None, mask=None):
        # Initial convolution and pooling
        x_token = self.pool(self.conv(tf.expand_dims(inputs, axis=-1)))

        # Add positional encoding and apply dropout
        positions = tf.range(start=0, limit=tf.shape(x_token)[1], delta=1)
        embedding = x_token + self.position(positions)

        if training:
            mask = tf.random.uniform(shape=tf.shape(embedding)[:2]) < self.masking_portion
            embedding = tf.where(mask[:, :, tf.newaxis], 0.0, embedding)

        x = self.dropout(embedding, training=training)

        # Pass through transformer encoder blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)

        # Deconvolution and final dense layer for each timestep in sequence
        x = self.deconv(x)
        y_pred = self.output_layer(x)  # Shape: (batch_size, window_size, 1)
        return y_pred  # Return the prediction for each timestep in the sequence

    def train_step(self, data):
        # Unpack the data
        x, y = data

        # Forward pass with GradientTape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Calculate gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Check for NaNs in gradients using check_numerics
        for i, grad in enumerate(gradients):
            if grad is not None:
                try:
                    tf.debugging.check_numerics(grad,
                                                f"NaN or Inf detected in gradient of variable {self.trainable_variables[i].name}")
                except tf.errors.InvalidArgumentError:
                    tf.print(f"NaN or Inf detected in gradient of variable {self.trainable_variables[i].name}")
                    raise

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}





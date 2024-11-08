import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

from custom_loss import bert4nilm_loss


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
            padding='SAME'
        )

        weighted_pooled = pooled * self.weight

        # Adding epsilon to avoid NaNs in the sqrt operation
        return tf.sqrt(weighted_pooled + self.epsilon)


class BERT4NILM(Model):
    def __init__(self, wandb_config):
        super(BERT4NILM, self).__init__()
        self.max_power = wandb_config.max_power
        self.on_threshold = wandb_config.on_threshold
        self.args = wandb_config

        # Model configuration parameters
        self.original_len = wandb_config.window_size
        self.latent_len = wandb_config.window_size
        self.dropout_rate = wandb_config.dropout
        self.hidden = wandb_config.head_size
        self.heads = wandb_config.num_heads
        self.n_transformer_blocks = wandb_config.n_layers
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
            self.build_transformer_block() for _ in range(self.n_transformer_blocks)
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

        # Calculate dynamic latent length after pooling
        sequence_length = tf.shape(x_token)[1]  # Get the actual sequence length after pooling
        self.latent_len = sequence_length  # Dynamically assign to latent_len

        # Positional encoding and dropout
        positions = tf.range(start=0, limit=self.latent_len, delta=1)
        positional_embedding = self.position(positions)
        embedding = x_token + positional_embedding

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
        return y_pred

    def appliance_state(self, y_true, y_pred):
        """
        Determines appliance state based on power consumption thresholds.

        Args:
            y_true (tensor): Ground truth power consumption values.
            y_pred (tensor): Predicted power consumption values.

        Returns:
            s_true (tensor): Ground truth appliance state labels {-1, 1}.
            s_pred (tensor): Predicted appliance state labels {-1, 1}.
        """
        # Clamp ground truth and predicted values to avoid noise affecting state detection
        max_power = self.max_power  # or set this based on appliance-specific max power if available
        y_true = tf.clip_by_value(y_true, 0, max_power)
        y_pred = tf.clip_by_value(y_pred, 0, max_power)

        # Use the on-threshold to determine state; anything above on_threshold is considered "on"
        s_true = tf.cast(y_true > self.on_threshold, dtype=tf.float32) * 2 - 1  # {-1, 1}
        s_pred = tf.cast(y_pred > self.on_threshold, dtype=tf.float32) * 2 - 1  # {-1, 1}

        return s_true, s_pred

    def train_step(self, data):
        # Unpack the data
        x, y_true = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Get the appliance state labels and predictions
            s_true, s_pred = self.appliance_state(y_true, y_pred)

            # Calculate the custom loss
            loss = bert4nilm_loss((y_true, y_pred), (s_true, s_pred))

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # monitor possible shape differences and force reshaping
        if tf.shape(y_true) != tf.shape(y_pred):
            y_pred = tf.reshape(y_pred, tf.shape(y_true))

        # Update and calculate metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # Collect all metrics to return
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = loss

        return metrics

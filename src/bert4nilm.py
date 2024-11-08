import tensorflow as tf
import wandb
from tensorflow.keras import layers, Model, regularizers

from custom_loss import bert4nilm_loss

tf.config.run_functions_eagerly(True)  # Forces eager execution in tf.function


class LearnedL2NormPooling(layers.Layer):
    """
    Learned L2 norm pooling layer, reduces sequence length by half while applying a learned weight to each channel.
    """

    def __init__(self, pool_size=2, epsilon=1e-6, **kwargs):
        super(LearnedL2NormPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.epsilon = epsilon  # Small value to prevent NaN issues

    def build(self, input_shape):
        # Adding a trainable weight parameter for each feature channel
        self.weight = self.add_weight(
            name='l2_norm_weight',
            shape=(1, 1, input_shape[-1]),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # Ensure correct dimensions by squeezing if needed
        if inputs.shape.ndims == 4:
            inputs = tf.squeeze(inputs, axis=-2)

        # Apply squared pooling
        squared_inputs = tf.square(inputs)

        # Average pooling with stability epsilon to prevent NaNs
        pooled = tf.nn.avg_pool1d(
            squared_inputs,
            ksize=self.pool_size,
            strides=self.pool_size,
            padding='SAME'
        )

        # Apply learned weight to the pooled output
        weighted_pooled = pooled * self.weight

        # Return the square root to approximate L2 norm with stability epsilon
        return tf.sqrt(weighted_pooled + self.epsilon)


class BERT4NILM(Model):
    """
    A custom model for NILM (Non-Intrusive Load Monitoring) based on the BERT architecture, with
    convolutional feature extraction, learned L2 norm pooling, positional embedding, and transformer blocks.
    """

    def __init__(self, wandb_config):
        super(BERT4NILM, self).__init__()
        # Configuration parameters
        self.batch_size = wandb_config.batch_size
        self.max_power = wandb_config.max_power
        self.on_threshold = wandb_config.on_threshold
        self.args = wandb_config

        # Model hyperparameters
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

        # Initial convolutional layer to extract features and increase hidden size
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

        # Learned L2 norm pooling layer to reduce sequence length by half
        self.pool = LearnedL2NormPooling()

        # Positional embeddings for each position in the sequence after pooling
        self.position = layers.Embedding(
            input_dim=self.latent_len,  # Input length post-pooling
            output_dim=self.hidden,
            embeddings_initializer=self.kernel_initializer
        )

        # Dropout layer for regularization
        self.dropout = layers.Dropout(self.dropout_rate)

        # Transformer encoder blocks for feature extraction and attention-based encoding
        self.transformer_blocks = [
            self.build_transformer_block() for _ in range(self.n_transformer_blocks)
        ]

        # Deconvolution layer to upsample the sequence length back to the original
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

        # Final dense layer for each timestep's output
        self.output_layer = layers.Dense(
            1,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

    @staticmethod
    def get_regularizer(regularizer_config):
        # Returns appropriate regularization based on the configuration
        if regularizer_config == 'l1':
            return regularizers.l1(l=0.01)
        elif regularizer_config == 'l2':
            return regularizers.l2(l=0.01)
        elif regularizer_config == 'l1_l2':
            return regularizers.l1_l2(l1=0.01, l2=0.01)
        else:
            return None

    def build_transformer_block(self):
        """
        Constructs a transformer encoder block with multi-head self-attention and feed-forward layers.
        """
        inputs = layers.Input(shape=(None, self.hidden))

        # Multi-head self-attention layer
        attn_output = layers.MultiHeadAttention(
            num_heads=self.heads,
            key_dim=self.hidden // self.heads
        )(inputs, inputs)
        attn_output = layers.Dropout(self.dropout_rate)(attn_output)
        attn_output = layers.Add()([inputs, attn_output])
        attn_output = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)(attn_output)

        # Feed-forward network with dense layers
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
        # Step 1: Convolutional layer to expand feature space
        x_token = self.pool(self.conv(tf.expand_dims(inputs, axis=-1)))

        # Step 2: Calculate sequence length after pooling
        sequence_length = tf.shape(x_token)[1]  # Adjust latent_len for actual sequence length post-pooling
        self.latent_len = sequence_length

        # Step 3: Positional embedding and addition
        positions = tf.range(start=0, limit=self.latent_len, delta=1)
        positional_embedding = self.position(positions)
        embedding = x_token + positional_embedding

        # Step 4: Masking random elements during training for regularization
        if training:
            mask = tf.random.uniform(shape=tf.shape(embedding)[:2]) < self.masking_portion
            embedding = tf.where(mask[:, :, tf.newaxis], 0.0, embedding)

        # Apply dropout to the embedding
        x = self.dropout(embedding, training=training)

        # Step 5: Transformer blocks for encoding
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)

        # Step 6: Deconvolution to restore original sequence length
        x = self.deconv(x)

        # Step 7: Final dense layer for output
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
        max_power = self.max_power
        y_true = tf.clip_by_value(y_true, 0, max_power)
        y_pred = tf.clip_by_value(y_pred, 0, max_power)

        # Use the on-threshold to determine state; values above threshold considered "on"
        s_true = tf.cast(y_true > self.on_threshold, dtype=tf.float32) * 2 - 1
        s_pred = tf.cast(y_pred > self.on_threshold, dtype=tf.float32) * 2 - 1

        return s_true, s_pred

    def train_step(self, data):
        # Unpack the data
        x, y_true = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Ensure shapes match before proceeding
            try:
                y_true_elements = tf.size(y_true)
                y_pred_elements = tf.size(y_pred)
                tf.debugging.assert_equal(
                    y_true_elements, y_pred_elements,
                    message="Shape mismatch: y_true and y_pred must have the same number of elements"
                )
                y_pred = tf.reshape(y_pred, tf.shape(y_true))

                # Get the appliance state labels and predictions
                s_true, s_pred = self.appliance_state(y_true, y_pred)

                # Calculate the custom loss
                loss = bert4nilm_loss((y_true, y_pred), (s_true, s_pred))

            except tf.errors.InvalidArgumentError as e:
                tf.print("Error in train_step:", e)
                return {}

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Ensure shapes match
        y_true = tf.ensure_shape(y_true, [None, None, 1])
        y_pred = tf.ensure_shape(y_pred, [None, None, 1])

        # Update and calculate metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # Collect all metrics to return
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = loss

        # Log to wandb at the specified interval
        if int(self.optimizer.iterations) % self.batch_size == 0:
            wandb.log({"loss": loss.numpy()})

        return metrics

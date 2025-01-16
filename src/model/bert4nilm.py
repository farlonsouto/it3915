import keras.layers
import tensorflow as tf
from tensorflow.keras import Model, layers

from .custom.layers import LearnedL2NormPooling, TransformerBlock


class BERT4NILM(Model):
    def __init__(self, config, is_training):
        super(BERT4NILM, self).__init__()
        self.config = config
        self.is_training = is_training

        # Embedding module with convolutional layer
        self.conv1d = layers.Conv1D(
            filters=config['hidden_size'],
            kernel_size=config['conv_kernel_size'],
            strides=config['conv_strides'],
            padding='same',
            activation=config['conv_activation'],
            kernel_regularizer="l1_l2"
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

        # Add initial layer norm and dropout (to match PyTorch)
        self.initial_layer_norm = layers.LayerNormalization(epsilon=config['layer_norm_epsilon'])
        self.initial_dropout = layers.Dropout(config['dropout'])

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

        # DeConv layer
        self.deconv = layers.Conv1DTranspose(
            filters=config['hidden_size'],
            kernel_size=config['deconv_kernel_size'],
            strides=config['deconv_strides'],
            padding='same',
            activation=config['deconv_activation'],
            kernel_regularizer="l1_l2"
        )

        self.dense1 = layers.Dense(config['hidden_size'], activation='linear')
        self.dense2 = layers.Dense(config['output_size'], activation='tanh')

    def call(self, inputs, training=False, mask=None):
        tf.debugging.check_numerics(inputs, 'inputs contains NaNs values')

        # Embedding module
        x = self.conv1d(inputs)  # The inputs already have the shape (batch_size, window_size, 1)

        x = self.l2_pool(x)

        # Add positional embedding
        x_token = x
        embedding = x_token + self.pos_embedding

        # Initial layer norm and dropout (matching the original PyTorch impl)
        x = self.initial_layer_norm(embedding)
        x = self.initial_dropout(x, training=training or self.is_training)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)

        # Permute back for deconv (matching PyTorch's permute operations)
        x = self.deconv(x)

        # Output module with tanh activation in between
        x = self.dense1(x)

        x = self.dense2(x)

        x = tf.math.scalar_mul(self.config['appliance_max_power'], x)

        return x

    def train_step(self, data):
        aggregated, y_true, mask = data

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(aggregated, training=True)

            # Adjust mask to match y_true/pred dimensions if necessary
            if len(mask.shape) < len(y_true.shape):
                mask = tf.expand_dims(mask, axis=-1)

            # Apply mask to calculate loss based only on the masked positions
            bool_mask = tf.cast(mask, tf.bool)
            y_true_masked = tf.boolean_mask(y_true, bool_mask)
            predictions_masked = tf.boolean_mask(predictions, bool_mask)

            # Compute loss
            loss = self.compiled_loss(y_true_masked, predictions_masked)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y_true_masked, predictions_masked)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}

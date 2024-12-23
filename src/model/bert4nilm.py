import tensorflow as tf
from tensorflow.keras import Model, layers

from src.custom.loss.bert4nilm import LossFunction
from .custom.layers import LearnedL2NormPooling, TransformerBlock


class BERT4NILM(Model):

    def __init__(self, config, is_training):
        super(BERT4NILM, self).__init__()
        self.config = config
        self.loss_function = LossFunction(config)

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

    # @tf.autograph.experimental.do_not_convert
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
        # Unpack the data
        aggregated, y_true, mask = data

        try:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = self(aggregated, training=True)
                # Compute loss with explicit mask parameter
                loss = self.loss_function(y_true, predictions, mask)

            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)

            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Update metrics
            self.compiled_metrics.update_state(y_true, predictions)

            # Update metrics (includes the metric that tracks the loss)
            for metric in self.metrics:
                if metric.name == "loss":
                    metric.update_state(loss)
                else:
                    metric.update_state(y_true, predictions)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}
        except Exception as e:
            print(f"Error in train_step: {e}")
            return {m.name: m.result() for m in self.metrics}

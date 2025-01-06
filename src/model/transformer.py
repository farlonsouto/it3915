import tensorflow as tf
from tensorflow.keras import Model, layers

from .custom.layers import PowerAwareAttention


class Transformer(Model):
    def __init__(self, config, is_training):
        super(Transformer, self).__init__()
        self.config = config
        self.is_training = is_training

        # Input embedding layer
        self.input_embedding = layers.Dense(config['hidden_size'])

        # Positional encoding
        self.pos_embedding = self.add_weight(
            "pos_embedding",
            shape=[config['window_size'], config['hidden_size']],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )

        # Initial normalization and dropout
        self.layer_norm1 = layers.LayerNormalization(epsilon=config['layer_norm_epsilon'])
        self.dropout1 = layers.Dropout(config['dropout'])

        # Power-aware attention layers and regular transformer components
        self.power_attention_layers = []
        self.ff_layers = []
        self.layer_norms1 = []
        self.layer_norms2 = []

        for _ in range(config['num_layers']):
            # Power-aware attention
            self.power_attention_layers.append(
                PowerAwareAttention(
                    num_heads=config['num_heads'],
                    key_dim=config['hidden_size'] // config['num_heads'],
                    value_dim=config['hidden_size'] // config['num_heads'],
                    power_threshold=config['on_threshold'],
                    dropout=config['dropout']
                )
            )

            # Feed-forward network
            self.ff_layers.append([
                layers.Dense(config['ff_dim'], activation="relu"),
                layers.Dropout(config['dropout']),
                layers.Dense(config['hidden_size'])
            ])

            # Layer normalizations
            self.layer_norms1.append(layers.LayerNormalization(epsilon=config['layer_norm_epsilon']))
            self.layer_norms2.append(layers.LayerNormalization(epsilon=config['layer_norm_epsilon']))

        # Output layers
        self.final_layer_norm = layers.LayerNormalization(epsilon=config['layer_norm_epsilon'])
        self.dense1 = layers.Dense(config['hidden_size'], activation='tanh')
        self.dense2 = layers.Dense(1)  # Output a single value per time step

    def call(self, inputs, training=False, mask=None, return_attention_weights=False):
        # Store attention weights for analysis
        all_attention_weights = []

        # Initial embedding
        x = self.input_embedding(inputs)
        x += self.pos_embedding

        # Initial normalization and dropout
        x = self.layer_norm1(x)
        x = self.dropout1(x, training=training or self.is_training)

        # Transformer blocks with power-aware attention
        for i in range(len(self.power_attention_layers)):
            attn_output, attn_weights = self.power_attention_layers[i](
                x, x, x,
                attention_mask=mask,
                training=training,
                return_attention_scores=True
            )
            x = self.layer_norms1[i](x + attn_output)
            all_attention_weights.append(attn_weights)

            # Feed-forward network
            ff_output = x
            for layer in self.ff_layers[i]:
                ff_output = layer(ff_output)
            x = self.layer_norms2[i](x + ff_output)

        # Output processing
        x = self.final_layer_norm(x)
        x = self.dense1(x)
        x = self.dense2(x)  # Output shape: [batch_size, seq_len, 1]

        # Return attention weights only if explicitly requested
        if return_attention_weights:
            return x, all_attention_weights
        return x

    def train_step(self, data):
        aggregated, y_true, mask = data

        with tf.GradientTape() as tape:
            predictions, attention_weights = self(aggregated, training=True)

            bool_mask = tf.cast(mask, tf.bool)
            y_true_masked = tf.boolean_mask(y_true, bool_mask)
            predictions_masked = tf.boolean_mask(predictions, bool_mask)

            # Calculate main loss
            reconstruction_loss = self.compiled_loss(y_true_masked, predictions_masked)

            # Add event detection loss
            event_loss = 0.0
            for attn_weights in attention_weights:
                # Calculate power differences
                power_diff = tf.abs(y_true[:, 1:, :] - y_true[:, :-1, :])

                # Create event mask
                event_mask = tf.cast(power_diff > self.config['on_threshold'], tf.float32)
                event_mask = tf.expand_dims(event_mask, axis=1)  # [batch_size, 1, seq_len, 1]

                # Adjust the shape for broadcasting
                event_mask = tf.expand_dims(event_mask, axis=-1)  # [batch_size, 1, seq_len, 1, 1]

                # Broadcast event_mask to match attn_weights shape
                broadcasted_mask = event_mask[:, :, :, :, 0]  # [batch_size, 1, seq_len, 1]

                # Ensure attention weights have the same length
                attn_weights_trimmed = attn_weights[:, :, :-1, :]  # [batch_size, num_heads, seq_len-1, seq_len]

                # Compute event detection loss using broadcasting
                diff = tf.square(broadcasted_mask - attn_weights_trimmed)
                event_loss += tf.reduce_mean(diff)

            # Combine losses
            total_loss = reconstruction_loss + self.config.get('event_loss_weight', 0.1) * event_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y_true_masked, predictions_masked)
        return {m.name: m.result() for m in self.metrics}

import tensorflow as tf
import wandb
from tensorflow.keras import layers, Model, regularizers

tf.config.run_functions_eagerly(True)  # Forces eager execution in tf.function


class LearnedL2NormPooling(layers.Layer):
    def __init__(self, pool_size=2, epsilon=1e-6, **kwargs):
        super(LearnedL2NormPooling, self).__init__(**kwargs)
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
        self.batch_size = wandb_config.batch_size
        self.args = wandb_config
        self.original_len = wandb_config.window_size
        self.latent_len = wandb_config.window_size // 2  # Adjusted for pooling
        self.dropout_rate = wandb_config.dropout
        self.hidden_size = wandb_config.hidden_size
        self.num_heads = wandb_config.num_heads
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
        self.max_power = wandb_config.max_power
        self.on_threshold = wandb_config.on_threshold

        self.conv = layers.Conv1D(
            filters=self.hidden_size,
            kernel_size=self.conv_kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        self.pool = LearnedL2NormPooling()
        self.position = layers.Embedding(
            input_dim=self.latent_len,
            output_dim=self.hidden_size,
            embeddings_initializer=self.kernel_initializer
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        self.transformer_blocks = [
            self.build_transformer_block() for _ in range(self.n_transformer_blocks)
        ]
        self.deconv = layers.Conv1DTranspose(
            filters=self.hidden_size,
            kernel_size=self.deconv_kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        self.output_layer1 = layers.Dense(
            self.hidden_size,
            activation='tanh',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        self.output_layer2 = layers.Dense(
            1,
            activation='sigmoid',  # To ensure output is between 0 and 1
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
        inputs = layers.Input(shape=(None, self.hidden_size))
        x = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)(inputs)
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_size // self.num_heads
        )(x, x)
        attn_output = self.dropout(attn_output)
        out1 = layers.Add()([inputs, attn_output])

        x = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)(out1)
        ff_output = layers.Dense(
            self.ff_dim,
            activation='gelu',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )(x)
        ff_output = layers.Dense(
            self.hidden_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )(ff_output)
        ff_output = self.dropout(ff_output)
        return Model(inputs, layers.Add()([out1, ff_output]))

    def call(self, inputs, training=None, mask=None):
        x_token = self.pool(self.conv(tf.expand_dims(inputs, axis=-1)))
        sequence_length = tf.shape(x_token)[1]
        self.latent_len = sequence_length
        positions = tf.range(start=0, limit=self.latent_len, delta=1)
        positional_embedding = self.position(positions)
        embedding = x_token + positional_embedding

        if training:
            mask = tf.random.uniform(shape=tf.shape(embedding)[:2]) < self.masking_portion
            embedding = tf.where(mask[:, :, tf.newaxis], 0.0, embedding)

        x = self.dropout(embedding, training=training)

        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)

        x = self.deconv(x)
        x = self.output_layer1(x)
        pred_appl_power = self.output_layer2(x)

        # Multiply by max power and clamp
        pred_appl_power = tf.clip_by_value(pred_appl_power * self.max_power, 0, self.max_power)
        print("Predicted appliance power: ", pred_appl_power)
        return pred_appl_power

    def train_step(self, data):
        inputs, targets = data

        with tf.GradientTape() as tape:
            pred_appl_power = self(inputs, training=True)
            loss = self.compiled_loss(targets, pred_appl_power)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(targets, pred_appl_power)
        metrics = {m.name: m.result().numpy() for m in self.metrics}
        metrics["loss"] = loss.numpy()

        if int(self.optimizer.iterations) % self.batch_size == 0:
            wandb.log(metrics)

        return metrics

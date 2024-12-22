from keras import layers, Model


class Seq2PointNILM(Model):
    def __init__(self, wandb_config):
        super(Seq2PointNILM, self).__init__()
        self.hyper_param = wandb_config

        # Convolutional layers
        self.conv1 = layers.Conv1D(
            filters=30,
            kernel_size=10,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        self.conv2 = layers.Conv1D(
            filters=30,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        self.conv3 = layers.Conv1D(
            filters=40,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        self.conv4 = layers.Conv1D(
            filters=50,
            kernel_size=5,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        self.conv5 = layers.Conv1D(
            filters=50,
            kernel_size=5,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        # Fully connected layers
        self.dense1 = layers.Dense(
            units=1024,
            activation='relu',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        self.dense2 = layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

    def call(self, inputs, training=None, mask=None):
        # Pass through convolutional layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Flatten and pass through dense layers
        x = layers.Flatten()(x)
        x = self.dense1(x)
        pred_appl_power = self.dense2(x)

        # Scale and clip predictions
        # pred_appl_power = tf.clip_by_value(pred_appl_power * self.hyper_param.appliance_max_power, 1.0,
        #                                   self.hyper_param.appliance_max_power)
        return pred_appl_power

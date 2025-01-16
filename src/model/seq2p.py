from keras import layers, Model


class Seq2PointNILM(Model):
    """
    Sequence-to-point learning with neural networks for non-intrusive load monitoring
    Chaoyun Zhang1 , Mingjun Zhong2 , Zongzuo Wang1 , Nigel Goddard1 , and Charles Sutton1
    arXiv:1612.09106v3 [stat.AP] 18 Sep 2017
    School of Informatics, University of Edinburgh, United Kingdom
    School of Computer Science, University of Lincoln, United Kingdom
    """

    def __init__(self, wandb_config):
        super(Seq2PointNILM, self).__init__()
        self.hyper_param = wandb_config

        self.kernel_regularizer = None
        if self.hyper_param.kernel_regularizer != 'None':
            self.kernel_regularizer = self.hyper_param.kernel_regularizer

        # Convolutional layers
        self.conv1 = layers.Conv1D(
            filters=30,
            kernel_size=10,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.kernel_regularizer
        )

        self.conv2 = layers.Conv1D(
            filters=30,
            kernel_size=8,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.kernel_regularizer
        )

        self.conv3 = layers.Conv1D(
            filters=40,
            kernel_size=6,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.kernel_regularizer
        )

        self.conv4 = layers.Conv1D(
            filters=50,
            kernel_size=5,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.kernel_regularizer
        )

        self.conv5 = layers.Conv1D(
            filters=50,
            kernel_size=5,
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.kernel_regularizer
        )

        # Flatten layer
        self.flatten = layers.Flatten()

        # Fully connected layers
        self.dense1 = layers.Dense(
            units=1024,
            activation='relu',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.kernel_regularizer
        )

        self.dense2 = layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.kernel_regularizer
        )

    def call(self, inputs, training=None, mask=None):
        # Pass through convolutional layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        pred_appl_power = self.dense2(x)

        return pred_appl_power

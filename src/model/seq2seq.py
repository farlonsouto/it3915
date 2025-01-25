from tensorflow.keras import layers, Model
from tensorflow.python.keras.layers import LSTM


class Seq2SeqNILM(Model):
    """
        2018 - The 6th IEEE International Conference on Smart Energy Grid Engineering
        Regularized LSTM Based Deep Learning Model: First Step towards Real-Time Non-Intrusive Load Monitoring
        Hasan Rafiq, Hengxu Zhang, Huimin Li, Manesh Kumar Ochani
        School of Electrical Engineering - Shandong University - Jinan, China
    """

    def __init__(self, wandb_config):
        super(Seq2SeqNILM, self).__init__()
        self.hyper_param = wandb_config

        # 1D Convolution layer for initial feature extraction
        self.conv1d = layers.Conv1D(
            filters=16,
            kernel_size=4,
            strides=1,
            padding='same',
            activation='linear',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.hyper_param.kernel_regularizer
        )

        # Input layer with dropout
        self.input_dropout = layers.Dropout(rate=self.hyper_param.dropout)

        # First Bidirectional LSTM layer with 128 units
        self.lstm1 = layers.Bidirectional(
            LSTM(
                units=128,
                return_sequences=True,
                kernel_initializer='truncated_normal',
                kernel_regularizer=self.hyper_param.kernel_regularizer,
            ),
            merge_mode='concat'
        )

        # Dropout between LSTM layers
        self.inter_dropout = layers.Dropout(rate=self.hyper_param.dropout)

        # Second Bidirectional LSTM layer with 256 units
        self.lstm2 = layers.Bidirectional(
            LSTM(
                units=256,
                return_sequences=True,
                kernel_initializer='truncated_normal',
                kernel_regularizer=self.hyper_param.kernel_regularizer
            ),
            merge_mode='concat'
        )

        # First dense layer with 256 units
        self.dense1 = layers.Dense(
            128,
            activation='tanh',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.hyper_param.kernel_regularizer
        )

        # Output dense layer
        self.output_layer = layers.Dense(
            1,
            activation='linear',
            kernel_initializer='truncated_normal',
            kernel_regularizer=self.hyper_param.kernel_regularizer
        )

    def call(self, inputs, training=None, mask=None):
        # Apply 1D convolution
        x = self.conv1d(inputs)

        # Apply input dropout
        x = self.input_dropout(x, training=training)

        # First Bidirectional LSTM layer
        x = self.lstm1(x)

        # Inter-layer dropout
        x = self.inter_dropout(x, training=training)

        # Second Bidirectional LSTM layer
        x = self.lstm2(x)

        # Dense layer
        x = self.dense1(x)

        # Output predictions
        pred_appl_power = self.output_layer(x)

        return pred_appl_power

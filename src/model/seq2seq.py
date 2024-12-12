import tensorflow as tf
from keras import layers, Model


class Seq2SeqNILM(Model):
    def __init__(self, wandb_config):
        super(Seq2SeqNILM, self).__init__()
        self.hyper_param = wandb_config

        # Encoder LSTM
        self.encoder_lstm = layers.LSTM(
            units=self.hyper_param.hidden_size,
            return_state=True,
            return_sequences=True,
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        # Decoder LSTM
        self.decoder_lstm = layers.LSTM(
            units=self.hyper_param.hidden_size,
            return_sequences=True,
            return_state=True,
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

        # Dense layer to predict appliance power
        self.output_layer = layers.Dense(
            1,
            activation='linear',
            kernel_initializer='truncated_normal',
            kernel_regularizer='l2'
        )

    def reverse_sequence(self, inputs):
        """Reverses the input sequence along the time axis."""
        return tf.reverse(inputs, axis=[1])

    def call(self, inputs, training=None, mask=None):
        # Reverse the input sequence for the encoder
        reversed_inputs = self.reverse_sequence(inputs)

        # Encoder
        encoder_outputs, state_h, state_c = self.encoder_lstm(reversed_inputs)

        # Decoder, initialized with the encoder's final states
        decoder_outputs, _, _ = self.decoder_lstm(encoder_outputs, initial_state=[state_h, state_c])

        # Output predictions
        pred_appl_power = self.output_layer(decoder_outputs)

        # Scale and clip predictions
        pred_appl_power = tf.clip_by_value(pred_appl_power * self.hyper_param.max_power, 1.0,
                                           self.hyper_param.max_power)
        return pred_appl_power

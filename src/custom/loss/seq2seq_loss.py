import tensorflow.keras.backend as k
from tensorflow.keras.losses import Loss


class Seq2SeqLoss(Loss):
    def __init__(self, name='seq2seq_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Implementation of equation (1) from the paper.

        Args:
            y_true: Target sequences of shape (batch_size, sequence_length)
            y_pred: Predicted sequences of shape (batch_size, sequence_length)

        Returns:
            Mean of the negative log likelihood across the batch and sequence
        """
        # The paper models this as a factorized Gaussian distribution
        # For a Gaussian with fixed variance, minimizing negative log likelihood
        # is equivalent to minimizing MSE
        return k.mean(k.square(y_pred - y_true))

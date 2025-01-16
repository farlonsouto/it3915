import tensorflow.keras.backend as k
from tensorflow.keras.losses import Loss


class Seq2PointLoss(Loss):
    def __init__(self, name='seq2point_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Implementation of equation (2) from the paper.

        Args:
            y_true: Target midpoint values of shape (batch_size, 1)
            y_pred: Predicted midpoint values of shape (batch_size, 1)

        Returns:
            Mean of the negative log likelihood across the batch
        """
        # The paper models this as a Gaussian distribution
        # For a Gaussian with fixed variance, minimizing negative log likelihood
        # is equivalent to minimizing MSE
        return k.mean(k.square(y_pred - y_true))

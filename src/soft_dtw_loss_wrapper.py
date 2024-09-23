import numpy as np
import tensorflow as tf
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


class DynamicTimeWarping(tf.keras.losses.Loss):
    def __init__(self, gamma=1.0, **kwargs):
        super(DynamicTimeWarping, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Use tf.py_function to apply the NumPy-based DTW in TensorFlow
        loss = tf.py_function(self.compute_dtw_loss, [y_true, y_pred], tf.float32)
        loss.set_shape([])  # Ensure the loss returns a scalar
        return loss

    def compute_dtw_loss(self, y_true, y_pred):
        # Convert TensorFlow tensors to NumPy arrays for SoftDTW
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        # Ensure both arrays are 2D
        y_true_np = np.atleast_2d(y_true_np)
        y_pred_np = np.atleast_2d(y_pred_np)

        # Compute the distance matrix using SquaredEuclidean
        D = SquaredEuclidean(y_true_np, y_pred_np)

        # Initialize SoftDTW and compute the loss
        sdtw = SoftDTW(D, gamma=self.gamma)
        loss = sdtw.compute()

        return np.float32(loss)

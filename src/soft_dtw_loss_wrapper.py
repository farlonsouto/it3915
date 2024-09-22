import numpy as np
import tensorflow as tf
from sdtw import SoftDTW  # Adjust to your SoftDTW package path


class SoftDTWLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.0):
        super(SoftDTWLoss, self).__init__()
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Ensure y_true and y_pred are at least 2D
        y_true = tf.ensure_shape(y_true, [None, None])  # Ensure at least 2D shape
        y_pred = tf.ensure_shape(y_pred, [None, None])  # Ensure at least 2D shape

        # Wrap the soft-DTW computation inside a tf.py_function
        def compute_soft_dtw(y_true_np, y_pred_np):
            # Compute distance matrix (Euclidean)
            D = np.linalg.norm(y_true_np[:, None] - y_pred_np[None, :], axis=-1)

            # Compute Soft-DTW using the NumPy implementation
            sdtw = SoftDTW(D, gamma=self.gamma)
            loss = sdtw.compute()
            return np.array([loss], dtype=np.float32)

        # Wrap the NumPy computation using tf.py_function
        loss = tf.py_function(func=compute_soft_dtw, inp=[y_true, y_pred], Tout=tf.float32)

        return loss

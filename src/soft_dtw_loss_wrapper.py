import numpy as np
import tensorflow as tf
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


class DynamicTimeWarping(tf.keras.losses.Loss):
    def __init__(self, gamma=1.0, **kwargs):
        super(DynamicTimeWarping, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        loss = tf.py_function(self.compute_dtw_loss, [y_true, y_pred], tf.float32)
        return loss

    def compute_dtw_loss(self, y_true, y_pred):
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        batch_size = y_true_np.shape[0]
        loss = 0

        for i in range(batch_size):
            D = SquaredEuclidean(y_true_np[i:i + 1], y_pred_np[i:i + 1])
            sdtw = SoftDTW(D, gamma=self.gamma)
            loss += sdtw.compute()

        return np.float32(loss / batch_size)

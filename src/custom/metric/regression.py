import tensorflow as tf


class MeanRelativeError(tf.keras.metrics.Metric):
    def __init__(self, name="MRE", **kwargs):
        super(MeanRelativeError, self).__init__(name=name, **kwargs)
        self.mre = self.add_weight(name="MRE", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the MRE state using the formula:
        MRE = mean(|y_true - y_pred| / max(y_true, y_pred, 1e-9))
        """
        # Compute the absolute error
        absolute_error = tf.abs(y_true - y_pred)

        # Compute the normalization term (element-wise max)
        normalization_term = tf.maximum(tf.maximum(y_true, y_pred), tf.keras.backend.epsilon())

        # Compute the relative error
        relative_error = absolute_error / normalization_term

        # Compute the mean relative error over all samples
        mre = tf.reduce_mean(relative_error)

        # Update the state
        self.mre.assign(mre)

    def result(self):
        """Return the current value of MRE."""
        return self.mre

    def reset_state(self):
        """Reset the metric state."""
        self.mre.assign(0.0)

import tensorflow as tf


class MeanRelativeError(tf.keras.metrics.Metric):
    def __init__(self, name="MRE", **kwargs):
        super(MeanRelativeError, self).__init__(name=name, **kwargs)
        self.total_mre = self.add_weight(name="total_mre", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true and y_pred have the same shape
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Compute the absolute error
        absolute_error = tf.abs(y_true - y_pred)

        # Compute the normalization term (element-wise max)
        normalization_term = tf.maximum(tf.maximum(tf.abs(y_true), tf.abs(y_pred)), 1e-9)

        # Compute the relative error
        relative_error = absolute_error / normalization_term

        # If sample_weight is provided, apply it
        if sample_weight is not None:
            relative_error = tf.multiply(relative_error, sample_weight)
            count = tf.reduce_sum(sample_weight)
        else:
            count = tf.cast(tf.size(y_true), tf.float32)

        # Compute the sum of relative errors
        total_relative_error = tf.reduce_sum(relative_error)

        # Update the total MRE and count
        self.total_mre.assign_add(total_relative_error)
        self.count.assign_add(count)

    def result(self):
        return self.total_mre / self.count

    def reset_state(self):
        self.total_mre.assign(0.0)
        self.count.assign(0.0)

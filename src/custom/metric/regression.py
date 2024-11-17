import tensorflow as tf


class MeanRelativeError(tf.keras.metrics.Metric):
    def __init__(self, name="MRE", **kwargs):
        super(MeanRelativeError, self).__init__(name=name, **kwargs)
        self.mre = self.add_weight(name="MRE", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mre = tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.reduce_mean(y_true) + 1e-7)))
        self.mre.assign(mre)

    def result(self):
        return self.mre

    def reset_states(self):
        self.mre.assign(0.0)

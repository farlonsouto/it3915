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


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, on_threshold, name='F1', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.on_threshold = on_threshold
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert power measurements to binary values
        y_true_binary = tf.cast(y_true > self.on_threshold, tf.float32)
        y_pred_binary = tf.cast(y_pred > self.on_threshold, tf.float32)

        self.precision.update_state(y_true_binary, y_pred_binary, sample_weight)
        self.recall.update_state(y_true_binary, y_pred_binary, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, on_threshold, name='Acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.on_threshold = on_threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = tf.cast(y_true > self.on_threshold, tf.float32)
        y_pred_binary = tf.cast(y_pred > self.on_threshold, tf.float32)

        correct_predictions = tf.equal(y_true_binary, y_pred_binary)
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(correct_predictions, tf.float32)))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.true_positives / self.total

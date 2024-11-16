import tensorflow as tf


class MREMetric(tf.keras.metrics.Metric):
    def __init__(self, name="mre", **kwargs):
        super(MREMetric, self).__init__(name=name, **kwargs)
        self.mre = self.add_weight(name="mre", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mre = tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.reduce_mean(y_true) + 1e-7)))
        self.mre.assign(mre)

    def result(self):
        return self.mre

    def reset_states(self):
        self.mre.assign(0.0)


class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, on_threshold, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.on_threshold = on_threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > self.on_threshold, tf.float32)
        y_pred = tf.cast(y_pred > self.on_threshold, tf.float32)

        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class AccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, on_threshold, name='acc', **kwargs):
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

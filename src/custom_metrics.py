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
    def __init__(self, name="f1_score", **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="true_positives", initializer="zeros")
        self.predicted_positives = self.add_weight(name="predicted_positives", initializer="zeros")
        self.actual_positives = self.add_weight(name="actual_positives", initializer="zeros")
        self.epsilon = 1e-10  # Small constant for numerical stability

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast to float32 and apply thresholding
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold = tf.reduce_mean(y_true) * 0.1
        y_true = tf.cast(y_true > threshold, tf.float32)
        y_pred = tf.cast(y_pred > threshold, tf.float32)

        # Calculate true positives, predicted positives, and actual positives
        true_positives = tf.reduce_sum(y_true * y_pred)
        predicted_positives = tf.reduce_sum(y_pred)
        actual_positives = tf.reduce_sum(y_true)

        # Update metric state
        self.true_positives.assign_add(true_positives)
        self.predicted_positives.assign_add(predicted_positives)
        self.actual_positives.assign_add(actual_positives)

    def result(self):
        # Calculate precision and recall, then F1 score
        precision = self.true_positives / (self.predicted_positives + self.epsilon)
        recall = self.true_positives / (self.actual_positives + self.epsilon)
        f1 = 2 * ((precision * recall) / (precision + recall + self.epsilon))
        return tf.clip_by_value(f1, 0.0, 1.0)

    def reset_states(self):
        # Reset metric state variables
        self.true_positives.assign(0.0)
        self.predicted_positives.assign(0.0)
        self.actual_positives.assign(0.0)


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

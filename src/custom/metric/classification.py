import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, on_threshold, name='F1', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.on_threshold = on_threshold
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.f1_score = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = tf.cast(y_true > self.on_threshold, tf.float32)
        y_pred_binary = tf.cast(y_pred > self.on_threshold, tf.float32)

        self.precision.update_state(y_true_binary, y_pred_binary, sample_weight)
        self.recall.update_state(y_true_binary, y_pred_binary, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()

        f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        self.f1_score.assign(f1)
        return self.f1_score

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
        self.f1_score.assign(0.)


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, on_threshold, name='Acc', **kwargs):
        """
        Initializes the Accuracy metric with a specified on_threshold value and other optional parameters.

        Args:
            on_threshold: float, the power value above which the appliance is considered "on".
            name: str, the name of the metric. Defaults to 'Acc'.
            **kwargs: additional keyword arguments to be passed to the parent class
                (tf.keras.metrics.Metric).

        Returns:
            Accuracy instance with initialized weights for tracking true positives and total samples.
        """
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

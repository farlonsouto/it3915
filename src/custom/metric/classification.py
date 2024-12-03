import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, on_threshold=0.5, name='F1', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.on_threshold = on_threshold

        # Initialize cumulative counters for TP, FP, FN
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update state variables for TP, FP, FN based on the current batch.
        """
        # Binarize predictions and labels based on the threshold
        y_true_binary = tf.cast(y_true >= self.on_threshold, tf.float32)
        y_pred_binary = tf.cast(y_pred >= self.on_threshold, tf.float32)

        # Calculate TP, FP, FN
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))

        # Update the cumulative counters
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """
        Calculate the F1 score based on cumulative TP, FP, FN.
        """
        # Calculate precision and recall
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        """
        Reset all state variables to zero.
        """
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


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

import tensorflow as tf


class Binarize:
    """
    Binarize predictions and labels based on a given threshold. Values above the threshold are considered positive.
    """

    def __call__(self, threshold, y_true, y_pred):
        y_true_binary = tf.cast(y_true >= threshold, tf.float32)
        y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
        return y_true_binary, y_pred_binary


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, on_threshold=0.5, name='F1', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.on_threshold = on_threshold
        self.recall_metric = tf.keras.metrics.Recall()
        self.precision_metric = tf.keras.metrics.Precision()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary, y_pred_binary = Binarize()(self.on_threshold, y_true, y_pred)
        self.recall_metric.update_state(y_true_binary, y_pred_binary)
        self.precision_metric.update_state(y_true_binary, y_pred_binary)

    def result(self):
        # Calculate precision and recall
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()

        if precision + recall == 0:
            print("------------------------------ WARNING: Precision + recall == 0. Using backend.epsilon()")
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.recall_metric.reset_state()
        self.precision_metric.reset_state()


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, on_threshold, name='Acc', **kwargs):
        """
        Accuracy metric with a specified threshold for binary classification.
        """
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.on_threshold = on_threshold
        self.true_positives_metric = tf.keras.metrics.TruePositives()
        self.true_negatives_metric = tf.keras.metrics.TrueNegatives()
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary, y_pred_binary = Binarize()(self.on_threshold, y_true, y_pred)
        self.true_positives_metric.update_state(y_true_binary, y_pred_binary)
        self.true_negatives_metric.update_state(y_true_binary, y_pred_binary)
        self.total_samples.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return (self.true_positives_metric.result() + self.true_negatives_metric.result()) / self.total_samples

    def reset_state(self):
        self.true_positives_metric.reset_state()
        self.true_negatives_metric.reset_state()
        self.total_samples.assign(0.0)

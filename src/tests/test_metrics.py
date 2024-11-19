import tensorflow as tf

from ..custom.metric.classification import F1Score, Accuracy
from ..custom.metric.regression import MeanRelativeError


class TestMetrics(tf.test.TestCase):
    def test_f1_score_all_zeros(self):
        f1 = F1Score(on_threshold=0.5)
        y_true = tf.constant([0.0, 0.0, 0.0, 0.0])
        y_pred = tf.constant([0.0, 0.0, 0.0, 0.0])

        f1.update_state(y_true, y_pred)
        self.assertAllClose(f1.result(), 0.0)  # Precision and recall both 0 -> F1 should be 0
        f1.reset_states()

    def test_f1_score_one_pred_nonzero_above_threshold(self):
        f1 = F1Score(on_threshold=0.5)
        y_true = tf.constant([0.0, 0.0, 0.0, 0.0])
        y_pred = tf.constant([0.0, 0.0, 0.0, 1.0])

        f1.update_state(y_true, y_pred)
        self.assertAllClose(f1.result(), 0.0)  # Precision and recall both 0 -> F1 should be 0
        f1.reset_states()

    def test_f1_score_all_nonzeros_below_threshold(self):
        f1 = F1Score(on_threshold=0.5)
        y_true = tf.constant([0.1, 0.2, 0.3])
        y_pred = tf.constant([0.1, 0.2, 0.3])

        f1.update_state(y_true, y_pred)
        self.assertAllClose(f1.result(), 0.0)  # Perfect match -> F1 should be 1.0
        f1.reset_states()

    def test_accuracy_all_zeros(self):
        acc = Accuracy(on_threshold=0.5)
        y_true = tf.constant([0.0, 0.0, 0.0])
        y_pred = tf.constant([0.0, 0.0, 0.0])

        acc.update_state(y_true, y_pred)
        self.assertAllClose(acc.result(), 1.0)  # Perfect match -> Accuracy should be 1.0
        acc.reset_states()

    def test_accuracy_small_values(self):
        acc = Accuracy(on_threshold=0.5)
        y_true = tf.constant([0.1, 0.2, 0.3])
        y_pred = tf.constant([0.1, 0.2, 0.3])

        acc.update_state(y_true, y_pred)
        self.assertAllClose(acc.result(), 1.0)  # All below threshold -> Accuracy should be 1.0
        acc.reset_states()

    def test_mean_relative_error_all_zeros(self):
        mre = MeanRelativeError()
        y_true = tf.constant([0.0, 0.0, 0.0])
        y_pred = tf.constant([0.0, 0.0, 0.0])

        mre.update_state(y_true, y_pred)
        # Edge case, avoiding division by zero, expect MRE = 0
        self.assertAllClose(mre.result(), 0.0)
        mre.reset_states()

    def test_mean_relative_error_small_values(self):
        mre = MeanRelativeError()
        y_true = tf.constant([0.1, 0.2, 0.3])
        y_pred = tf.constant([0.11, 0.19, 0.31])

        mre.update_state(y_true, y_pred)
        # Small deviations in predictions, expect nonzero small MRE
        self.assertAllClose(mre.result(), 0.05, atol=1e-2)  # Adjust the tolerance for small numbers
        mre.reset_states()

    def test_f1_score_non_zeros_above(self):
        f1 = F1Score(on_threshold=0.5)
        y_true = tf.constant([0.6, 0.0, 0.7])
        y_pred = tf.constant([0.7, 0.0, 0.8])

        f1.update_state(y_true, y_pred)
        self.assertAllClose(f1.result(), 1.0)  # Precision and recall are perfect
        f1.reset_states()

    def test_accuracy_edge_case(self):
        acc = Accuracy(on_threshold=0.5)
        y_true = tf.constant([0.6, 0.0, 0.7])
        y_pred = tf.constant([0.7, 0.0, 0.8])

        acc.update_state(y_true, y_pred)
        self.assertAllClose(acc.result(), 1.0)  # Perfect match -> Accuracy should be 1.0
        acc.reset_states()

    def test_mean_relative_error_edge_case(self):
        mre = MeanRelativeError()
        y_true = tf.constant([0.6, 0.0, 0.7])
        y_pred = tf.constant([0.7, 0.0, 0.8])

        mre.update_state(y_true, y_pred)
        # Small relative errors in predictions, expect nonzero small MRE
        self.assertGreater(mre.result(), 0.0)
        mre.reset_states()


# Run the tests
if __name__ == "__main__":
    tf.test.main()

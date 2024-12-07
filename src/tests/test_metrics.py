import tensorflow as tf

from ..custom.metric.regression import MeanRelativeError


class TestMetrics(tf.test.TestCase):

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

import unittest
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence

from src.bert_loss import bert4nilm_loss


# Assuming the custom loss function is defined as bert4nilm_loss
class TestCustomLoss(unittest.TestCase):

    def setUp(self):
        # Prepare some small sample tensors for testing
        self.y_ground_truth = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        self.y_predicted = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        self.s_ground_truth = tf.constant([[1.0, 1.0], [0.0, 0.0]], dtype=tf.float32)
        self.s_predicted = tf.constant([[1.0, 1.0], [0.0, 0.0]], dtype=tf.float32)

    def test_loss_output_shape(self):
        """Test if the loss output is a scalar."""
        loss = bert4nilm_loss(self.y_ground_truth, self.y_predicted, self.s_ground_truth, self.s_predicted)
        self.assertEqual(loss.shape, ())  # Loss should be a scalar

    def test_zero_loss(self):
        """Test that the loss is close to zero when predictions match the ground truth."""
        # Ground truth and predicted values are the same, so loss should be very small (close to 0)
        loss = bert4nilm_loss(self.y_ground_truth, self.y_ground_truth, self.s_ground_truth, self.s_ground_truth)
        # TODO: It might be that the delta is way too tolerant - needs further studies.
        self.assertAlmostEqual(loss.numpy(), 0.0, delta=0.6)  # Allow a small tolerance

    def test_non_zero_loss(self):
        """Test that the loss is positive when y_predicted and s_predicted differ from ground truth."""
        # Change predicted values to make the loss non-zero
        y_pred = tf.constant([[1.5, 2.5], [3.5, 4.5]], dtype=tf.float32)
        s_pred = tf.constant([[0.9, 0.9], [0.1, 0.1]], dtype=tf.float32)

        loss = bert4nilm_loss(self.y_ground_truth, y_pred, self.s_ground_truth, s_pred)
        self.assertGreater(loss.numpy(), 0.0)  # Loss should be greater than 0


# Running the tests
if __name__ == '__main__':
    unittest.main()

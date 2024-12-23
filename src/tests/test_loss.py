import unittest

import numpy as np
import tensorflow as tf

from ..custom.loss.bert4nilm import LossFunction  # Adjust this import to match your file structure


class TestLossFunction(unittest.TestCase):
    def setUp(self):
        # Configuration for the loss function
        class Config:
            temperature = 1.0
            on_threshold = 100.0
            lambda_val = 0.1

        self.loss_function = LossFunction(Config())

    def test_partial_mask(self):
        # Test case where 25% of the mask is True
        batch_size = 64
        window_size = 240

        y_true = tf.random.uniform((batch_size, window_size, 1), minval=0.0, maxval=3000.0)
        y_pred = tf.random.uniform((batch_size, window_size, 1), minval=0.0, maxval=3000.0)

        # Create a mask where 25% of the values are True
        mask = np.zeros((batch_size, window_size), dtype=bool)
        indices = np.random.choice(window_size, size=window_size // 4, replace=False)
        mask[:, indices] = True
        mask = tf.convert_to_tensor(mask, dtype=tf.bool)

        # Compute loss
        loss = self.loss_function(y_true, y_pred, mask)

        # Assert that loss is a scalar
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.shape, ())

        # Assert loss is non-negative
        self.assertGreaterEqual(loss.numpy(), 0.0)

    def test_full_mask(self):
        # Test case where the mask is fully True
        batch_size = 64
        window_size = 240

        y_true = tf.random.uniform((batch_size, window_size, 1), minval=0.0, maxval=3000.0)
        y_pred = tf.random.uniform((batch_size, window_size, 1), minval=0.0, maxval=3000.0)

        # Create a mask where all values are True
        mask = tf.ones((batch_size, window_size), dtype=tf.bool)

        # Compute loss
        loss = self.loss_function(y_true, y_pred, mask)

        # Assert that loss is a scalar
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.shape, ())

        # Assert loss is non-negative
        self.assertGreaterEqual(loss.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()

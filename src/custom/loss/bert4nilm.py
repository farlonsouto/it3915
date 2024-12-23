import tensorflow as tf


class LossFunction:
    def __init__(self, config):
        self.temperature = float(config.temperature)
        self.on_threshold = int(config.on_threshold)
        self.lambda_val = float(config.lambda_val)

    def __call__(self, y_true, y_pred, mask):
        """
        Compute masked loss without modifying `y_true` or `y_pred`.

        Args:
            y_true: Ground truth values, shape (batch_size, window_size, features).
            y_pred: Predicted values, shape (batch_size, window_size, features).
            mask: Boolean mask with shape (batch_size, window_size).
                  True indicates positions to compute loss for.

        Returns:
            Scalar loss value.
        """
        eps = 1e-7

        # Ensure the mask is float32 for mathematical operations
        mask = tf.cast(mask, tf.float32)

        # Reshape mask to match y_true and y_pred
        mask = tf.expand_dims(mask, axis=-1)  # Shape: (batch_size, window_size, 1)

        # MSE term
        mse = tf.reduce_sum(tf.square(y_pred - y_true) * mask) / (tf.reduce_sum(mask) + eps)

        # KL divergence term
        y_true_dist = tf.nn.softmax(y_true / self.temperature + eps, axis=-1)
        y_pred_dist = tf.nn.softmax(y_pred / self.temperature + eps, axis=-1)
        kl_div = tf.reduce_sum(
            tf.keras.losses.KLDivergence()(y_true_dist, y_pred_dist) * mask
        ) / (tf.reduce_sum(mask) + eps)

        # Binary cross-entropy term
        status_true = tf.cast(y_true > self.on_threshold, tf.float32)
        status_pred = tf.clip_by_value(tf.cast(y_pred > self.on_threshold, tf.float32), eps, 1.0 - eps)
        bce = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(status_true, status_pred) * mask
        ) / (tf.reduce_sum(mask) + eps)

        # L1 term
        l1 = tf.reduce_sum(tf.abs(y_pred - y_true) * mask) / (tf.reduce_sum(mask) + eps)

        # Total loss
        total_loss = mse + 0.1 * kl_div + 0.1 * bce + 0.01 * self.lambda_val * l1
        return total_loss

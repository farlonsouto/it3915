import tensorflow as tf
from tensorflow.keras.losses import Loss


class LossFunction(Loss):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(config.temperature)
        self.on_threshold = int(config.on_threshold)
        self.lambda_val = float(config.lambda_val)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred, mask=None):
        """
        Computes masked loss considering only positions where mask=True.

        Args:
            y_true: Target values with shape (batch_size, window_size, features)
            y_pred: Predicted values with shape (batch_size, window_size, features)
            mask: Boolean mask with shape (batch_size, window_size, 1)
                 True indicates positions to compute loss for
        """
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, -1e7, 1e7)

        # Convert mask to float32 for multiplication
        mask = tf.cast(mask, tf.float32)

        # Print debug info
        tf.print("\nMask statistics:")
        tf.print("Sum of mask (number of positions to compute loss for):", tf.reduce_sum(mask))
        tf.print("Mask shape:", tf.shape(mask))
        tf.print("Percentage of positions used for loss:",
                 tf.reduce_mean(tf.cast(mask > 0, tf.float32)) * 100, "%")

        tf.print("\nInput shapes:")
        tf.print("y_true shape:", tf.shape(y_true))
        tf.print("y_pred shape:", tf.shape(y_pred))
        tf.print("mask shape:", tf.shape(mask))

        tf.print("\nValue ranges at masked positions (where mask=1):")
        tf.print("y_true values where mask=1:",
                 tf.boolean_mask(y_true, tf.cast(mask, tf.bool)))
        tf.print("y_pred values where mask=1:",
                 tf.boolean_mask(y_pred, tf.cast(mask, tf.bool)))

        # Apply mask to get only the positions we want to compute loss for
        y_pred_masked = y_pred * mask
        y_true_masked = y_true * mask

        # MSE term
        mse = tf.reduce_sum(tf.square(y_pred_masked - y_true_masked)) / (tf.reduce_sum(mask) + eps)
        tf.print("\nMSE term:", mse)

        # KL divergence
        y_true_dist = tf.nn.softmax(y_true_masked / self.temperature + eps, axis=-1)
        y_pred_dist = tf.nn.softmax(y_pred_masked / self.temperature + eps, axis=-1)
        kl_div = tf.reduce_sum(
            tf.keras.losses.KLDivergence()(y_true_dist, y_pred_dist)
        ) / (tf.reduce_sum(mask) + eps)
        tf.print("KL divergence term:", kl_div)

        # Binary cross-entropy
        status_true = tf.cast(y_true_masked > self.on_threshold, tf.float32)
        status_pred = tf.clip_by_value(tf.cast(y_pred_masked > self.on_threshold, tf.float32), eps, 1.0 - eps)
        bce = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(status_true, status_pred)
        ) / (tf.reduce_sum(mask) + eps)
        tf.print("BCE term:", bce)

        # L1 term
        l1 = tf.reduce_sum(tf.abs(y_pred_masked - y_true_masked)) / (tf.reduce_sum(mask) + eps)
        tf.print("L1 term:", l1)

        tf.print("\nValue ranges for masked positions:")
        tf.print("Max y_true_masked:", tf.reduce_max(y_true_masked))
        tf.print("Min y_true_masked:", tf.reduce_min(y_true_masked))
        tf.print("Max y_pred_masked:", tf.reduce_max(y_pred_masked))
        tf.print("Min y_pred_masked:", tf.reduce_min(y_pred_masked))

        # Total loss
        total_loss = mse + 0.1 * kl_div + 0.1 * bce + 0.01 * self.lambda_val * l1
        tf.print("\nTotal loss:", total_loss)

        return total_loss

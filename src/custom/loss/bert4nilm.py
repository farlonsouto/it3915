import tensorflow as tf
from tensorflow.keras.losses import Loss


# Updated LossFunction
class LossFunction(Loss):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.temperature = config.temperature
        self.on_threshold = config.on_threshold
        self.lambda_val = config.lambda_val
        self.masking_token = config.mask_token

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        # Add numerical stability
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, -1e7, 1e7)

        # Identify the masked portion using the masking_token
        mask = tf.reduce_all(tf.equal(y_true, self.masking_token), axis=-1)
        mask = tf.cast(mask, tf.float32)

        # Ensure y_true and y_pred have the same shape
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)

        tf.debugging.assert_equal(y_true_shape, y_pred_shape, "y_true and y_pred must have the same shape")

        # Apply mask to predictions and true values
        y_pred_masked = y_pred * tf.expand_dims(mask, axis=-1)
        y_true_masked = y_true * tf.expand_dims(mask, axis=-1)

        # MSE term
        mse = tf.reduce_sum(tf.square(y_pred_masked - y_true_masked)) / (tf.reduce_sum(mask) + eps)

        # KL divergence
        y_true_dist = tf.nn.softmax(y_true_masked / self.temperature + eps, axis=-1)
        y_pred_dist = tf.nn.softmax(y_pred_masked / self.temperature + eps, axis=-1)
        kl_div = tf.reduce_sum(
            tf.keras.losses.KLDivergence()(y_true_dist, y_pred_dist)
        ) / (tf.reduce_sum(mask) + eps)

        # Binary cross-entropy
        status_true = tf.cast(y_true_masked > self.on_threshold, tf.float32)
        status_pred = tf.clip_by_value(tf.cast(y_pred_masked > self.on_threshold, tf.float32), eps, 1.0 - eps)
        bce = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(status_true, status_pred)
        ) / (tf.reduce_sum(mask) + eps)

        # L1 term
        l1 = tf.reduce_sum(tf.abs(y_pred_masked - y_true_masked)) / (tf.reduce_sum(mask) + eps)

        # Total loss
        total_loss = mse + 0.1 * kl_div + 0.1 * bce + 0.01 * self.lambda_val * l1
        return tf.where(tf.math.is_finite(total_loss), total_loss, 1e3)

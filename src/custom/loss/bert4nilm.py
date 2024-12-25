from abc import ABC

import tensorflow as tf


class LossFunction(tf.keras.losses.Loss, ABC):
    def __init__(self, config):
        super().__init__()
        self.temperature = float(config.temperature)
        self.on_threshold = int(config.on_threshold)
        self.lambda_val = float(config.lambda_val)

    def __call__(self, y_true, y_pred, sample_weight=None):
        eps = 1e-7

        # MSE term
        mse = tf.reduce_mean(tf.square(y_pred - y_true))

        # KL divergence term
        y_true_dist = tf.nn.softmax(y_true / self.temperature + eps, axis=-1)
        y_pred_dist = tf.nn.softmax(y_pred / self.temperature + eps, axis=-1)
        kl_div = tf.reduce_mean(tf.keras.losses.KLDivergence()(y_true_dist, y_pred_dist))

        # Binary cross-entropy term
        status_true = tf.cast(y_true > self.on_threshold, tf.float32)
        status_pred = tf.clip_by_value(tf.cast(y_pred > self.on_threshold, tf.float32), eps, 1.0 - eps)
        bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(status_true, status_pred))

        # L1 term
        l1 = tf.reduce_mean(tf.abs(y_pred - y_true))

        # Debug prints
        # tf.print("\nDebug values:")
        # tf.print("MSE:", mse)
        # tf.print("KL:", kl_div)
        # tf.print("BCE:", bce)
        # tf.print("L1:", l1)

        total_loss = mse + 0.1 * kl_div + 0.1 * bce + 0.01 * self.lambda_val * l1
        return total_loss

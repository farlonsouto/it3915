import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class LossFunction(tf.keras.losses.Loss):
    """ Encapsulates a normalized variation of the loss function described in the paper BERT4NILM: A Bidirectional
    Transformer Model for Non-Intrusive Load Monitoring, Zhenrui Yue et. al"""

    def __init__(self, wandb_config, reduction=losses_utils.ReductionV2.AUTO, name="bert4nilm_loss"):
        super().__init__(reduction, name)
        self.max_power = wandb_config.max_power
        self.lambda_val = wandb_config.lambda_val
        self.mask_positions = None,
        self.is_training = False

    def call(self, app_pw_grd_truth, app_pw_predicted):
        """
        Compute the loss function.

        Arguments:
        - app_pw_grd_truth: Ground truth values.
        - app_pw_predicted: Predicted values.
        - mask_positions: Mask tensor indicating masked positions (1.0 for masked, 0.0 otherwise).
        - is_training: Boolean flag to switch between training and validation loss behavior.

        Returns:
        - Total loss value.
        """

        # Reshape predictions to match ground truth
        app_pw_predicted = tf.reshape(app_pw_predicted, tf.shape(app_pw_grd_truth))

        # Clip predictions to the valid range
        app_pw_predicted = tf.clip_by_value(app_pw_predicted, 0, self.max_power)

        tau = 1.0  # Softmax temperature

        # Mean Squared Error (MSE)
        mse_loss = tf.square(app_pw_predicted - app_pw_grd_truth)

        # Kullback-Leibler Divergence
        softmax_true = tf.nn.softmax(app_pw_grd_truth / tau)
        softmax_pred = tf.nn.softmax(app_pw_predicted / tau)
        kl_diverg = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(
            softmax_true, softmax_pred
        )

        # L1 Loss
        l1_loss = tf.abs(app_pw_predicted - app_pw_grd_truth)

        if self.is_training and self.mask_positions is not None:
            # Apply masking to all loss components
            mask = tf.cast(self.mask_positions, tf.float32)
            mse_loss = tf.reduce_sum(mse_loss * mask) / tf.reduce_sum(mask)
            kl_diverg = tf.reduce_sum(kl_diverg * mask) / tf.reduce_sum(mask)
            l1_loss = tf.reduce_sum(l1_loss * mask) / tf.reduce_sum(mask)
        else:
            # Use all positions during validation/testing
            mse_loss = tf.reduce_mean(mse_loss)
            kl_diverg = tf.reduce_mean(kl_diverg)
            l1_loss = tf.reduce_mean(l1_loss)

        # Normalize and combine losses
        norm_mse = mse_loss / (1 + mse_loss)
        norm_kl_diverg = kl_diverg / (1 + kl_diverg)
        norm_l1_loss = (l1_loss * self.lambda_val) / (1 + l1_loss * self.lambda_val)

        total_loss = norm_mse + norm_kl_diverg + norm_l1_loss

        return total_loss

import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class LossFunction():
    """ Encapsulates a normalized variation of the loss function described in the paper BERT4NILM: A Bidirectional
    Transformer Model for Non-Intrusive Load Monitoring, Zhenrui Yue et. al"""

    def __init__(self, wandb_config, reduction=losses_utils.ReductionV2.AUTO, name="bert4nilm_loss"):
        self.max_power = wandb_config.max_power
        self.lambda_val = wandb_config.lambda_val

    def compute(self, app_pw_grd_truth, app_pw_predicted, mask):  # Added mask parameter
        app_pw_predicted = tf.reshape(app_pw_predicted, tf.shape(app_pw_grd_truth))
        app_pw_predicted = tf.clip_by_value(app_pw_predicted, 0, self.max_power)

        # Compute Mean Squared Error (MSE)
        mse_loss = tf.square(app_pw_predicted - app_pw_grd_truth)

        # Compute L1 Loss
        l1_loss = tf.abs(app_pw_predicted - app_pw_grd_truth)

        # Apply mask
        mask = tf.cast(mask, tf.float32)  # Use input mask instead of mask_observer
        mse_loss = tf.reduce_sum(mse_loss * mask) / tf.reduce_sum(mask)
        l1_loss = tf.reduce_sum(l1_loss * mask) / tf.reduce_sum(mask)

        # Normalize and combine losses
        norm_mse = mse_loss / (1 + mse_loss)
        norm_l1_loss = (l1_loss * self.lambda_val) / (1 + l1_loss * self.lambda_val)

        total_loss = norm_mse + norm_l1_loss

        return total_loss

import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class LossFunction():
    """ Encapsulates a normalized variation of the loss function described in the paper BERT4NILM: A Bidirectional
    Transformer Model for Non-Intrusive Load Monitoring, Zhenrui Yue et. al"""

    def __init__(self, wandb_config, reduction=losses_utils.ReductionV2.AUTO, name="bert4nilm_loss"):
        self.max_power = wandb_config.max_power
        self.lambda_val = wandb_config.lambda_val

    def compute(self, app_pw_grd_truth, app_pw_predicted, mask_positions=None, is_training=False):

        # Debug shapes
        # tf.print("Ground truth shape:", tf.shape(app_pw_grd_truth))
        # tf.print("Predicted shape:", tf.shape(app_pw_predicted))

        # Reshape predictions to match ground truth
        app_pw_predicted = tf.reshape(app_pw_predicted, tf.shape(app_pw_grd_truth))
        #  tf.print("Reshaped predicted shape:", tf.shape(app_pw_predicted))

        # Clip predictions to the valid range
        app_pw_predicted = tf.clip_by_value(app_pw_predicted, 0, self.max_power)

        tau = 1.0  # Softmax temperature

        # Compute Mean Squared Error (MSE)
        mse_loss = tf.square(app_pw_predicted - app_pw_grd_truth)
        # tf.print("MSE loss shape:", tf.shape(mse_loss))

        # Compute KL-Divergence at sequence level
        softmax_true = tf.nn.softmax(app_pw_grd_truth / tau, axis=-2)  # [batch_size, seq_len, 1]
        softmax_pred = tf.nn.softmax(app_pw_predicted / tau, axis=-2)  # [batch_size, seq_len, 1]
        kl_diverg = softmax_true * tf.math.log(
            softmax_true / (softmax_pred + 1e-10) + 1e-10)  # [batch_size, seq_len, 1]
        kl_diverg = tf.reduce_sum(kl_diverg, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
        # tf.print("KL divergence shape:", tf.shape(kl_diverg))

        # Compute L1 Loss
        l1_loss = tf.abs(app_pw_predicted - app_pw_grd_truth)
        # tf.print("L1 loss shape:", tf.shape(l1_loss))

        # Apply mask if training
        if is_training and mask_positions is not None:
            mask = tf.cast(mask_positions['original'], tf.float32)  # [batch_size, seq_len, 1]
            # tf.print("Mask shape:", tf.shape(mask))

            mse_loss = tf.reduce_sum(mse_loss * mask) / tf.reduce_sum(mask)
            kl_diverg = tf.reduce_sum(kl_diverg * mask) / tf.reduce_sum(mask)
            l1_loss = tf.reduce_sum(l1_loss * mask) / tf.reduce_sum(mask)
        else:
            mse_loss = tf.reduce_mean(mse_loss)
            kl_diverg = tf.reduce_mean(kl_diverg)
            l1_loss = tf.reduce_mean(l1_loss)

        # Normalize and combine losses
        norm_mse = mse_loss / (1 + mse_loss)
        norm_kl_diverg = kl_diverg / (1 + kl_diverg)
        norm_l1_loss = (l1_loss * self.lambda_val) / (1 + l1_loss * self.lambda_val)

        total_loss = norm_mse + norm_kl_diverg + norm_l1_loss
        # tf.print("Total loss shape:", tf.shape(total_loss))

        return total_loss

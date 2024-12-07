import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class LossFunction(tf.keras.losses.Loss):
    """ Encapsulates a normalized variation of the loss function described in the paper BERT4NILM: A Bidirectional
    Transformer Model for Non-Intrusive Load Monitoring, Zhenrui Yue et. al"""

    def __init__(self, wandb_config, reduction=losses_utils.ReductionV2.AUTO, name="bert4nilm_loss"):
        super().__init__(reduction, name)
        self.max_power = wandb_config.max_power
        self.lambda_val = wandb_config.lambda_val

    def call(self, app_pw_grd_truth, app_pw_predicted):
        """
        Custom loss function based on the equation provided by the paper 'BERT4NILM: A Bidirectional Transformer Model for
        Non-Intrusive Load Monitoring - Zhenrui Yue et. al'.
,
        Arguments:
            app_pw_grd_truth -- Ground truth values, the actual readings at the appliance meter.
            app_pw_predicted -- Predicted values, the model's output.

        Returns:
        loss -- The computed loss value.
        """

        # Assigning values and ensuring same shapes
        app_pw_predicted = tf.reshape(app_pw_predicted, tf.shape(app_pw_grd_truth))

        # Clip predictions to be within the valid range [0, max_power]
        app_pw_predicted = tf.clip_by_value(app_pw_predicted, 0, self.max_power)

        tau = 1.0

        # Mean Squared Error (MSE)
        mse_loss = tf.reduce_mean(tf.square(app_pw_predicted - app_pw_grd_truth))

        # Kullback-Leibler Divergence (KLD) between softmax distributions
        softmax_true = tf.nn.softmax(app_pw_grd_truth / tau)
        softmax_pred = tf.nn.softmax(app_pw_predicted / tau)

        # Original code for 1 GPU or 1 CPU
        # kl_loss = KLDivergence()(softmax_true, softmax_pred)

        # Code to handle reduction into the multiple GPUs scenario:
        # Modify the KL divergence calculation
        kl_diverg = tf.reduce_sum(
            tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(softmax_true, softmax_pred))

        # Log-sigmoid cross-entropy term deliberately removed because we don't care about the on/off state
        # log_sigmoid_loss = tf.reduce_mean(
        #    tf.math.log(1 + tf.exp(-app_on_off_state_predicted * app_on_off_state_grd_truth)))

        # L1-norm (MAE) over a subset O (assuming that the subset O corresponds to all the data points)
        l1_loss = tf.reduce_mean(tf.abs(app_pw_predicted - app_pw_grd_truth))
        l1_loss = l1_loss * self.lambda_val  # For some appliances it is meaningful. For the kettle lambdas just 1.

        # The original formula:
        # total_loss = mse_loss + kl_diverg + self.lambda_val * l1_loss

        norm_mse = (mse_loss / 1 + mse_loss)
        norm_l1_loss = (l1_loss / 1 + l1_loss)
        norm_kl_diverg = (kl_diverg / 1 + kl_diverg)
        # norm_log_sigm_loss = (log_sigmoid_loss / 1 + log_sigmoid_loss)

        # The suggested variation where the terms were normalized, without the term regarding the appliance state
        total_loss = norm_mse + norm_l1_loss + norm_kl_diverg

        return total_loss

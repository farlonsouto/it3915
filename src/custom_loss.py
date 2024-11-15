import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class Bert4NilmLoss(tf.keras.losses.Loss):
    """ Encapsulates a normalized variation of the loss function described in the paper BERT4NILM: A Bidirectional
    Transformer Model for Non-Intrusive Load Monitoring, Zhenrui Yue et. al"""

    def __init__(self, wandb_config, reduction=losses_utils.ReductionV2.AUTO, name="bert4nilm_loss"):
        super().__init__(reduction, name)
        self.max_power = wandb_config.max_power
        self.on_threshold = wandb_config.on_threshold

    def __appliance_state(self, y_true, y_pred):
        """
        Determines appliance state based on power consumption thresholds.

        Args:
            y_true (tensor): Ground truth power consumption values.
            y_pred (tensor): Predicted power consumption values.

        Returns:
            s_true (tensor): Ground truth appliance state labels {-1, 1}.
            s_pred (tensor): Predicted appliance state labels {-1, 1}.
        """
        # Clamp ground truth and predicted values to avoid noise affecting state detection
        max_power = self.max_power
        y_true = tf.clip_by_value(y_true, 0, max_power)
        y_pred = tf.clip_by_value(y_pred, 0, max_power)

        # Use the on-threshold to determine state; values above threshold considered "on"
        s_true = tf.cast(y_true > self.on_threshold, dtype=tf.float32) * 2 - 1
        s_pred = tf.cast(y_pred > self.on_threshold, dtype=tf.float32) * 2 - 1

        return s_true, s_pred

    def __bert4nilm_loss_computation(self, power_readings: tuple, state_readings: tuple):
        """
        Custom loss function as per the equation provided by the paper 'BERT4NILM: A Bidirectional Transformer Model for
        Non-Intrusive Load Monitoring - Zhenrui Yue et. al'.
,
        Arguments:
            power_readings -- tuple of (Ground truth values for x, Predicted values for x (model's output)).
            state_readings -- tuple of (Ground truth on/off state, Predicted on/off state).

        Returns:
        loss -- The computed loss value.
        """

        # Assigning values and ensuring same shapes
        app_pw_grd_truth, app_pw_predicted = power_readings
        app_pw_predicted = tf.reshape(app_pw_predicted, tf.shape(app_pw_grd_truth))

        # Clip predictions to be within the valid range [0, max_power]
        app_pw_predicted = tf.clip_by_value(app_pw_predicted, 0, self.max_power)

        app_on_off_state_grd_truth, app_on_off_state_predicted = state_readings
        app_on_off_state_predicted = tf.reshape(app_on_off_state_predicted, tf.shape(app_on_off_state_grd_truth))

        tau = 1.0
        # lambda_val = 0.1

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

        # Log-sigmoid cross-entropy term
        log_sigmoid_loss = tf.reduce_mean(
            tf.math.log(1 + tf.exp(-app_on_off_state_predicted * app_on_off_state_grd_truth)))

        # L1-norm (MAE) over a subset O (assuming the subset O is all data points here)
        l1_loss = tf.reduce_mean(tf.abs(app_pw_predicted - app_pw_grd_truth))

        # The complete original formula:
        # total_loss = mse_loss + kl_diverg + log_sigmoid_loss + lambda_val * l1_loss

        norm_mse = (mse_loss / 1 + mse_loss)
        norm_l1_loss = (l1_loss / 1 + l1_loss)
        norm_kl_diverg = (kl_diverg / 1 + kl_diverg)
        norm_log_sigm_loss = (log_sigmoid_loss / 1 + log_sigmoid_loss)

        # The suggested variation where the terms were normalized and the term regarding the appliance state domains
        # the function
        total_loss = norm_mse + norm_l1_loss + norm_kl_diverg + (3 * norm_log_sigm_loss)

        return total_loss

    def call(self, app_pw_grd_truth, app_pw_pred):
        return self.__bert4nilm_loss_computation((app_pw_grd_truth, app_pw_pred),
                                                 self.__appliance_state(app_pw_grd_truth, app_pw_pred))

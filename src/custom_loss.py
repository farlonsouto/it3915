import tensorflow as tf


def nde_loss(y_true, y_pred):
    """
    The Normalized Disaggregation Error (NDE) as (much as possible as) proposed in 'Deep Neural Networks Applied to
    Energy Disaggregation', by Jack Kelly and William Knottenbelt, published in 2015.
    """
    # Cast inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip predictions to prevent extreme values
    y_pred = tf.clip_by_value(y_pred, -1e6, 1e6)

    # Normalized Disaggregation Error
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true))
    nde = tf.sqrt(numerator / denominator)

    # Non-negative penalty with clipping
    non_negative_penalty = tf.reduce_mean(tf.maximum(-y_pred, 0))

    # Mean Squared Error with clipping
    mse = tf.reduce_mean(tf.square(tf.clip_by_value(y_true - y_pred, -1e6, 1e6)))

    # Combine the loss components with scaled weights
    total_loss = mse + 0.1 * nde + 0.01 * non_negative_penalty

    return tf.clip_by_value(total_loss, 0.0, 1e6)


def bert4nilm_loss(power_readings: tuple, state_readings: tuple):
    """
    Custom loss function as per the equation provided by the paper 'BERT4NILM: A Bidirectional Transformer Model for
    Non-Intrusive Load Monitoring - Zhenrui Yue et. al'.

    Arguments:
        power_readings -- tuple of (Ground truth values for x, Predicted values for x (model's output)).
        state_readings -- tuple of (Ground truth on/off state, Predicted on/off state).

    Returns:
    loss -- The computed loss value.
    """

    # Assigning values and ensuring same shapes
    app_pw_grd_truth, app_pw_predicted = power_readings
    app_pw_predicted = tf.reshape(app_pw_predicted, tf.shape(app_pw_grd_truth))

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
    log_sigmoid_loss = tf.reduce_mean(tf.math.log(1 + tf.exp(-app_on_off_state_predicted * app_on_off_state_grd_truth)))

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

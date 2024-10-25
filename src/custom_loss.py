import tensorflow as tf


def nde_loss(y_true, y_pred):
    """
    Modified loss function with better numerical stability
    """
    epsilon = 1e-10

    # Cast inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip predictions to prevent extreme values
    y_pred = tf.clip_by_value(y_pred, -1e6, 1e6)

    # Normalized Disaggregation Error
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true)) + epsilon
    nde = tf.sqrt(numerator / denominator)

    # Non-negative penalty with clipping
    non_negative_penalty = tf.reduce_mean(tf.maximum(-y_pred, 0))

    # Mean Squared Error with clipping
    mse = tf.reduce_mean(tf.square(tf.clip_by_value(y_true - y_pred, -1e6, 1e6)))

    # Combine the loss components with scaled weights
    total_loss = mse + 0.1 * nde + 0.01 * non_negative_penalty

    return tf.clip_by_value(total_loss, 0.0, 1e6)


def bert4nilm_loss(y_ground_truth, y_predicted, s_ground_truth, s_predicted, tau=1.0, lambda_val=0.1):
    """
    Custom loss function as per the equation provided by the paper 'BERT4NILM: A Bidirectional Transformer Model for
    Non-Intrusive Load Monitoring - Zhenrui Yue et. al'.

    Arguments:
    x_ground_truth -- Ground truth values for x.
    x_predicted -- Predicted values for x (output of the model).
    s_ground_truth -- Ground truth values for s (confidence or auxiliary predictions).
    s_predicted -- Predicted values for s (softmax outputs or logits).
    tau -- Temperature parameter for softmax (default is 1.0).
    lambda_val -- Regularization constant for the log-sigmoid term (default is 0.1).

    Returns:
    loss -- The computed loss value.
    """

    # Mean Squared Error (MSE)
    mse_loss = tf.reduce_mean(tf.square(y_predicted - y_ground_truth))

    # Kullback-Leibler Divergence between softmax distributions
    softmax_true = tf.nn.softmax(y_ground_truth / tau)
    softmax_pred = tf.nn.softmax(y_predicted / tau)
    kl_loss = KLDivergence()(softmax_true, softmax_pred)

    # Log-sigmoid cross-entropy term
    log_sigmoid_loss = tf.reduce_mean(tf.math.log(1 + tf.exp(-s_predicted * s_ground_truth)))

    # L1-norm (MAE) over a subset O (assuming the subset O is all data points here)
    l1_loss = tf.reduce_mean(tf.abs(y_predicted - y_ground_truth))

    # The complete formula
    total_loss = mse_loss + kl_loss + log_sigmoid_loss + lambda_val * l1_loss

    return total_loss

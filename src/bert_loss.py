import tensorflow as tf
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras import backend as K


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

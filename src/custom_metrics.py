import tensorflow as tf


def mre_metric(y_ground_truth, y_prediction):
    """
    Modified Mean Relative Error with better numerical stability
    """
    y_ground_truth = tf.cast(y_ground_truth, tf.float32)
    y_prediction = tf.cast(y_prediction, tf.float32)

    # Add small epsilon to denominator to prevent division by zero
    epsilon = 1e-10

    # Clip predictions to prevent extreme values
    y_prediction = tf.clip_by_value(y_prediction, -1e6, 1e6)

    relative_error = tf.abs(y_ground_truth - y_prediction) / (tf.abs(y_ground_truth) + epsilon)
    # Clip the relative error to prevent extreme values
    relative_error = tf.clip_by_value(relative_error, 0.0, 1e6)

    return tf.reduce_mean(relative_error)


def f1_score(y_ground_truth, y_prediction):
    """
    Modified F1 score with better numerical stability
    """
    epsilon = 1e-10

    # Ensure inputs are float32
    y_ground_truth = tf.cast(y_ground_truth, tf.float32)
    y_prediction = tf.cast(y_prediction, tf.float32)

    # Use a threshold relative to the data
    threshold = tf.reduce_mean(y_ground_truth) * 0.1

    y_ground_truth = tf.cast(y_ground_truth > threshold, tf.float32)
    y_prediction = tf.cast(y_prediction > threshold, tf.float32)

    true_positives = tf.reduce_sum(y_ground_truth * y_prediction)
    predicted_positives = tf.reduce_sum(y_prediction)
    actual_positives = tf.reduce_sum(y_ground_truth)

    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (actual_positives + epsilon)

    f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
    return tf.clip_by_value(f1, 0.0, 1.0)


def nde_metric(y_true, y_pred):
    """
    Modified Normalized Disaggregation Error with better numerical stability
    """
    epsilon = 1e-10

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip predictions to prevent extreme values
    y_pred = tf.clip_by_value(y_pred, -1e6, 1e6)

    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true)) + epsilon

    return tf.sqrt(tf.clip_by_value(numerator / denominator, 0.0, 1e6))

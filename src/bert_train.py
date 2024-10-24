import tensorflow as tf
from nilmtk import DataSet
from wandb.integration.keras import WandbCallback

import wandb
from bert4nilm import BERT4NILM
from bert_loss import bert4nilm_loss
from time_series_helper import TimeSeriesHelper


def sae_metric(y_ground_truth, y_prediction):
    """
    Calculates the Summed Absolute Error (SAE) between ground truth and predicted values.

    Args:
        y_ground_truth (Tensor): Ground truth values.
        y_prediction (Tensor): Predicted values.

    Returns:
        Tensor: Summed Absolute Error (SAE) between ground truth and predicted values.
    """
    y_ground_truth = tf.reshape(y_ground_truth, [-1])
    y_prediction = tf.reshape(y_prediction, [-1])
    return tf.abs(tf.reduce_sum(y_ground_truth) - tf.reduce_sum(y_prediction))


def mre_metric(y_ground_truth, y_prediction):
    """
    Calculates the Mean Relative Error (MRE) between ground truth and predicted values.

    The Mean Relative Error is the average absolute difference between the ground truth and predicted values,
    normalized by the ground truth value.

    Args:
        y_ground_truth (Tensor): Ground truth values.
        y_prediction (Tensor): Predicted values.

    Returns:
        Tensor: Mean Relative Error (MRE) between ground truth and predicted values.
    """
    y_ground_truth = tf.reshape(y_ground_truth, [-1])
    y_prediction = tf.reshape(y_prediction, [-1])
    relative_error = tf.abs(y_ground_truth - y_prediction) / (y_ground_truth + tf.keras.backend.epsilon())
    return tf.reduce_mean(relative_error)


def f1_score(y_ground_truth, y_prediction):
    """
    Calculates the F1 score between ground truth and predicted values.

    The F1 score is the harmonic mean of precision and recall, where an F1 score reaches its best value at 1 and worst at 0.

    Args:
        y_ground_truth (Tensor): Ground truth values.
        y_prediction (Tensor): Predicted values.

    Returns:
        Tensor: F1 score between ground truth and predicted values.

    Notes:
        This implementation assumes binary classification, where ground truth and predicted values are converted to binary (0/1) values.
    """
    y_ground_truth = tf.cast(y_ground_truth > 0, tf.float32)
    y_prediction = tf.cast(y_prediction > 0, tf.float32)

    true_positives = tf.reduce_sum(y_ground_truth * y_prediction)
    predicted_positives = tf.reduce_sum(y_prediction)
    actual_positives = tf.reduce_sum(y_ground_truth)

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())

    f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return f1


# Initialize WandB for tracking

# The article doesn't specify a single on_threshold value for all appliances. Instead, it mentions different
# on-thresholds for various appliances in Table 1. For example:

# - Fridge: 50W
# - Washer: 20W
# - Microwave: 200W
# - Dishwasher: 10W
# - Kettle: 2000W

# These values are specific to each appliance and are used to determine when an appliance is considered to be in the
# "on" state. In the context of the 'kettle' appliance that was used in the example, the correct on_threshold should be
# 2000W, not 50W. To correct this, we should modify the WandB configuration in the runner script to use the appropriate
# on_threshold for each appliance:

wandb.init(
    project="nilm_bert_transformer",
    config={
        "on_threshold": 50,
        "window_size": 128,
        "batch_size": 128,
        "head_size": 128,
        "num_heads": 2,
        "n_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "epochs": 10,
        "optimizer": "adam",
        "loss": "bert4nilm_loss",
        "tau": 1.0,
        "lambda_val": 0.1,
        "masking_portion": 0.25,
        "output_size": 1,
        "conv_kernel_size": 4,
        "deconv_kernel_size": 4,
        "embedding_dim": 128,
        "pooling_type": "max",  # Options: 'max', 'average'
        "conv_activation": "relu",
        "dense_activation": "tanh",
        "conv_filters": 128,  # Now separate from head_size
        "ff_dim": 512,  # Feed-forward network dimension
        "layer_norm_epsilon": 1e-6,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,  # Options: None, 'l1', 'l2', 'l1_l2'
        "bias_regularizer": None,  # Options: None, 'l1', 'l2', 'l1_l2'
    }
)

# Retrieve the configuration from WandB
wandb_config = wandb.config

# Load the NILMTK dataset
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-01', end='2015-02-15')

# Helper to preprocess time series data
timeSeriesHelper = TimeSeriesHelper(dataset, wandb_config.window_size, wandb_config.batch_size, appliance='kettle',
                                    on_threshold=wandb_config.on_threshold)

# Instantiate the BERT4NILM model
bert_model = BERT4NILM(wandb_config)

# Build the model by providing an input shape
# NOTICE: The 3D input_shape is (Batch size, window size, features) out of the time series. Where:
# `None` stands for a flexible, variable batch size.
# 'window_size` is the number of time steps
# `1` is the number of features (for now, only one: the power consumption)

bert_model.build((None, wandb_config.window_size, 1))

# Compile the model using the WandB configurations and the custom loss
optimizer = tf.keras.optimizers.Adam(learning_rate=wandb_config.learning_rate)


def custom_loss_wrapper(y_true, y_pred):
    # Split y_true and y_pred into their respective components
    # Assuming that y_true and y_pred are concatenated tensors
    # where the first half represents x and the second half represents s
    x_ground_truth, s_ground_truth = tf.split(y_true, num_or_size_splits=2, axis=-1)
    x_predicted, s_predicted = tf.split(y_pred, num_or_size_splits=2, axis=-1)

    return bert4nilm_loss(
        x_ground_truth, x_predicted,
        s_ground_truth=s_ground_truth,
        s_predicted=s_predicted,
        tau=wandb_config.tau,
        lambda_val=wandb_config.lambda_val
    )


# Use bert4nilm_loss from bert_loss.py, and pass any required arguments from wandb_config
# Compile the model
bert_model.compile(
    optimizer=optimizer,
    loss=custom_loss_wrapper,
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.MeanSquaredError(name='mse'),
        mre_metric,
        f1_score,
        sae_metric
    ]
)

# Print the model summary
bert_model.summary()

# Train the model and track the training process using WandB
history = bert_model.fit(
    timeSeriesHelper.getTrainingDataGenerator(),
    epochs=wandb_config.epochs,
    validation_data=timeSeriesHelper.getTestDataGenerator(),
    callbacks=[WandbCallback(monitor='val_loss', save_model=False)]
)

# Finish the WandB run
wandb.finish()

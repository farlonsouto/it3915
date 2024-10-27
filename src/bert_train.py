import numpy as np
import tensorflow as tf
import wandb
from nilmtk import DataSet
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from wandb.integration.keras import WandbCallback, WandbMetricsLogger

from bert4nilm import BERT4NILM
from custom_loss import bert4nilm_loss, nde_loss
from custom_metrics import mre_metric, f1_score, nde_metric
from time_series_helper import TimeSeriesHelper

# GPU memory to grow as needed. Tries to avoid GPU out-of-memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
        "loss": "nde_loss",
        # "loss": "bert4nilm_loss",
        "on_threshold": 2000,
        "window_size": 64,
        "batch_size": 128,
        "head_size": 64,
        "num_heads": 2,
        "n_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-5,
        "epochs": 2,
        "optimizer": "adam",
        "tau": 1.0,
        "lambda_val": 0.1,
        "masking_portion": 0.2,
        "output_size": 1,
        "conv_kernel_size": 8,
        "deconv_kernel_size": 8,
        "embedding_dim": 64,
        "pooling_type": "max",  # Options: 'max', 'average'
        "conv_activation": "relu",
        "dense_activation": "tanh",
        "conv_filters": 32,  # separate from head_size
        "ff_dim": 128,  # Feed-forward network dimension
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

# After creating the TimeSeriesHelper
# After creating your data generators
train_gen = timeSeriesHelper.getTrainingDataGenerator()
X_batch, y_batch = train_gen[0]
print("Sample statistics:")
print(f"X mean: {np.mean(X_batch)}, std: {np.std(X_batch)}")
print(f"y mean: {np.mean(y_batch)}, std: {np.std(y_batch)}")
print(f"X range: [{np.min(X_batch)}, {np.max(X_batch)}]")
print(f"y range: [{np.min(y_batch)}, {np.max(y_batch)}]")

# Ensure these shapes match
X_sample, y_sample = train_gen[0]
print(f"Sample batch shapes - X: {X_sample.shape}, y: {y_sample.shape}")
assert X_sample.shape == (wandb_config.batch_size, wandb_config.window_size, 1), "Incorrect input shape"
assert y_sample.shape == (wandb_config.batch_size, wandb_config.window_size, 1), "Incorrect target shape"

# Instantiate the BERT4NILM model
bert_model = BERT4NILM(wandb_config)

# Build the model by providing an input shape
# NOTICE: The 3D input_shape is (Batch size, window size, features) out of the time series. Where:
# `None` stands for a flexible, variable batch size.
# 'window_size` is the number of time steps
# `1` is the number of features (for now, only one: the power consumption)
# Build the model
bert_model.build((None, wandb_config.window_size, 1))

# Compile the model using the WandB configurations and the custom loss
optimizer = tf.keras.optimizers.Adam(
    learning_rate=wandb_config.learning_rate,
    clipnorm=1.0,  # gradient clipping
    clipvalue=0.5
)


# Mapping the loss function from WandB configuration to TensorFlow's predefined loss functions
loss_fn_mapping = {
    "mse": tf.keras.losses.MeanSquaredError(),
    "mae": tf.keras.losses.MeanAbsoluteError(),
    "nde_loss": nde_loss,  # Example of an additional loss function
}

# Get the loss function from the WandB config
loss_fn = loss_fn_mapping.get(wandb_config.loss, tf.keras.losses.MeanSquaredError())  # Default to MSE

# Use bert4nilm_loss from bert_loss.py, and pass any required arguments from wandb_config
# Compile the model
bert_model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.MeanSquaredError(name='mse'),
        mre_metric,
        f1_score,
        nde_metric
    ]
)

# Print the model summary
bert_model.summary()

my_callbacks = [
    WandbMetricsLogger(log_freq='epoch'),
    # , GradientDebugCallback()
    # , BatchStatsCallback()
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('../models/bert_model.keras', save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir='../logs')
]

# Train the model and track the training process using WandB
history = bert_model.fit(
    timeSeriesHelper.getTrainingDataGenerator(),
    epochs=wandb_config.epochs,
    validation_data=timeSeriesHelper.getTestDataGenerator(),
    callbacks=my_callbacks
)

# Finish the WandB run
wandb.finish()

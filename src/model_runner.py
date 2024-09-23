# Import necessary libraries
import random

import tensorflow as tf
from nilmtk import DataSet
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from soft_dtw_loss_wrapper import DynamicTimeWarping  # Import your DTW class
from time_series_helper import TimeSeriesHelper
from transformer import Transformer


# Custom SAE metric
def sae_metric(y_true, y_prediction):
    y_true = tf.reshape(y_true, [-1, 128])  # Ensure y_true has the correct shape
    y_prediction = tf.reshape(y_prediction, [-1, 128])  # Ensure y_pred has the correct shape
    # Proceed with the DTW computation

    # Reshape the tensors to make sure they have the right shape
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_prediction = tf.reshape(y_prediction, [tf.shape(y_prediction)[0], -1])

    # Calculate the sum of absolute error between true and predicted sums
    return tf.abs(tf.reduce_sum(y_true) - tf.reduce_sum(y_prediction))


# Start a run, tracking hyperparameters
wandb.init(
    project="nilm_transformer",
    config={
        "layer_1": 1024,
        "activation_1": "relu",
        "dropout": random.uniform(0.28, 0.45),
        "layer_2": 16,
        "activation_2": "softmax",
        "optimizer": "adam",
        "loss": "mean_squared_error",
        "metric": "mae",
        "epoch": 2,
        "batch_size": 512
    }
)

# Load data using NILMTK
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-01', end='2015-02-15')

# Define the window size and batch size
window_size = 128
batch_size = 512

timeSeriesHelper = TimeSeriesHelper(dataset, window_size, batch_size)

# Instantiate the Transformer model
transformer_model = Transformer((window_size, 1), head_size=32, num_heads=2, ff_dim=64, num_transformer_blocks=2,
                                dropout=0.3).create_transformer_model(metrics=['mae', sae_metric])

config = wandb.config

# Compile the model
transformer_model.compile(optimizer=config.optimizer,
                          # loss=DynamicTimeWarping(gamma=1.0),
                          loss=config.loss,
                          metrics=['mae'])

# Print model summary
transformer_model.summary()

# Get a batch of data to check shapes
x_batch, y_batch = next(iter(timeSeriesHelper.getTrainingDataGenerator()))
print(f"Input shape: {x_batch.shape}, Label shape: {y_batch.shape}")

# Train the model
history = transformer_model.fit(timeSeriesHelper.getTrainingDataGenerator(),
                                epochs=config.epoch,
                                batch_size=config.batch_size,
                                validation_data=timeSeriesHelper.getTestDataGenerator(),
                                callbacks=[
                                    WandbMetricsLogger(log_freq=5),
                                    WandbModelCheckpoint("../models")
                                ])

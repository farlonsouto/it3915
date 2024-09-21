# Import necessary libraries
import random

import numpy as np
import wandb
from nilmtk import DataSet
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from time_series_helper import TimeSeriesHelper
from transformer import Transformer

'''
NTNU IT3920 - Master Thesis - MSIT
Farlon de Alencar Souto
Transformer NN Architecture Applied to NILM - From a vanilla to an (auto) tuned version.
'''

# Start a run, tracking hyperparameters
wandb.init(
    # Set the wandb project where this run will be logged
    project="nilm_transformer",

    # Track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "dropout": random.uniform(0.1, 0.5),  # Adjusted dropout range
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "adam",  # Adam optimizer recommended for transformers
        "loss": "mean_squared_error",  # MSE is more appropriate for regression tasks
        "metric": "mae",  # MAE metric for regression tasks
        "epoch": 2,  # Set to 2 for testing memory and functionality first
        "batch_size": 256
    }
)

# [optional] use wandb.config as your config
config = wandb.config

# Load data using NILMTK
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-01', end='2015-02-15')

# Define the window size and batch size
window_size = 128  # Nearly 13min of data. Power of 2 allows for some computational advantages.
batch_size = 512  # Also to explore some computation advantages

timeSeriesHelper = TimeSeriesHelper(dataset, window_size, batch_size)

transformer_model = Transformer((window_size, 1), head_size=32, num_heads=2, ff_dim=64, num_transformer_blocks=2,
                                dropout=0.3).create_transformer_model()


# Custom SAE metric
def sae_metric(y_true, y_pred):
    return abs(np.sum(y_true) - np.sum(y_pred))


# Compile the model
transformer_model.compile(optimizer=config.optimizer,
                          loss='mean_absolute_error',
                          metrics=['mae', sae_metric]  # Adding SAE metric
                          )

# Print model summary
transformer_model.summary()

# Train the model
history = transformer_model.fit(timeSeriesHelper.getTrainingDataGenerator(),
                                epochs=config.epoch,
                                batch_size=config.batch_size,
                                validation_data=timeSeriesHelper.getTestDataGenerator(),
                                callbacks=[
                                    WandbMetricsLogger(log_freq=5),
                                    WandbModelCheckpoint("../models")
                                ])

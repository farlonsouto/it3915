import tensorflow as tf
from nilmtk import DataSet
from wandb.integration.keras import WandbCallback

import wandb
from soft_dtw_loss_wrapper import DynamicTimeWarping
from time_series_manager import TimeSeries
from transformer import Transformer


def sae_metric(y_ground_truth, y_prediction):
    y_ground_truth = tf.reshape(y_ground_truth, [-1])
    y_prediction = tf.reshape(y_prediction, [-1])
    return tf.abs(tf.reduce_sum(y_ground_truth) - tf.reduce_sum(y_prediction))


wandb.init(
    project="nilm_transformer",
    config={
        "window_size": 128,
        "batch_size": 512,
        "head_size": 32,
        "num_heads": 2,
        "ff_dim": 64,
        "num_transformer_blocks": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "epochs": 10,
        "optimizer": "adam",
        # loss": "dynamic_time_warping",
        "loss": "mean_absolute_error",
    }
)

config = wandb.config

# Load data using NILMTK
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-01', end='2015-02-15')

timeSeriesHelper = TimeSeries(dataset, config.window_size, config.batch_size)

# Instantiate the Transformer model
transformer_model = Transformer(
    (config.window_size, 1),
    head_size=config.head_size,
    num_heads=config.num_heads,
    ff_dim=config.ff_dim,
    num_transformer_blocks=config.num_transformer_blocks,
    dropout=config.dropout
).create_transformer_model()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
loss = DynamicTimeWarping(gamma=1.0) if config.loss == "dynamic_time_warping" else config.loss

transformer_model.compile(optimizer=optimizer, loss=loss, metrics=['mae', sae_metric])

# Print model summary
transformer_model.summary()

# Train the model
history = transformer_model.fit(
    timeSeriesHelper.getTrainingDataGenerator(),
    epochs=config.epochs,
    validation_data=timeSeriesHelper.getTestDataGenerator(),
    callbacks=[WandbCallback()]
)

wandb.finish()

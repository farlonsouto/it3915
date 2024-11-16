import numpy as np
import tensorflow as tf
import wandb
from nilmtk import DataSet
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger

from bert4nilm import BERT4NILM
from bert_wandb_init import config
from custom_loss import Bert4NilmLoss
from custom_metrics import MREMetric, F1ScoreMetric, AccuracyMetric
from gpu_memory_allocation import set_gpu_memory_growth
from time_series_uk_dale import TimeSeries


def create_model():
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
        "bert4nilm_loss": Bert4NilmLoss(wandb_config)
    }

    # Get the loss function from the WandB config
    loss_fn = loss_fn_mapping.get(wandb_config.loss, tf.keras.losses.MeanSquaredError())  # Default to MSE

    # Instantiate the BERT4NILM model
    model = BERT4NILM(wandb_config)

    # Build the model by providing an input shape
    # NOTICE: The 3D input_shape is (Batch size, window size, features) out of the time series. Where:
    # `None` stands for a flexible, variable batch size.
    # 'window_size` is the number of time steps
    # `1` corresponds the number of features (for now, only one: the power consumption)
    model.build((None, wandb_config.window_size, 1))

    # Use bert4nilm_loss from bert_loss.py, and pass any required arguments from wandb_config
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            AccuracyMetric(wandb_config.on_threshold),
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            MREMetric(),
            F1ScoreMetric(on_threshold=wandb_config.on_threshold)
        ]
    )

    return model


# Set GPU memory growth
set_gpu_memory_growth()

wandb.init(
    project="nilm_bert_transformer",
    entity="bert_training",
    config=config
)

# Retrieve the configuration from WandB
wandb_config = wandb.config

bert_model = create_model()

path_to_dataset = '../datasets/ukdale.h5'
print("Fetching data from the dataset located at ", path_to_dataset)
dataset = DataSet(path_to_dataset)

# time series handler for the UK Dale dataset
timeSeries = TimeSeries(dataset, [1, 3, 4, 5], [2], wandb_config)

# time series handler for the AMPds2dataset
# timeSeries = TimeSeries(dataset, wandb_config.window_size, wandb_config.batch_size,
#                        appliance=wandb_config.appliance)

train_gen = timeSeries.getTrainingDataGenerator()
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

print("... The training data is available. Starting training ...")

my_callbacks = [
    WandbMetricsLogger(log_freq='epoch'),
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('../models/bert_model', save_best_only=True, monitor='loss', save_format="tf")
]

# Train the model and track the training process using WandB
history = bert_model.fit(
    timeSeries.getTrainingDataGenerator(),
    epochs=wandb_config.epochs,
    validation_data=timeSeries.getTestDataGenerator(),
    callbacks=my_callbacks
)

# Finish the WandB run
wandb.finish()

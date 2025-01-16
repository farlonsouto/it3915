import numpy as np
import tensorflow as tf
import wandb
from nilmtk import DataSet
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from cmd_line_input import get_args
from custom.metric.regression import MeanRelativeError
from data.timeseries import TimeSeries
from gpu.gpu_memory_allocation import set_gpu_memory_growth
from hyper_params import for_model_appliance
from model.factory import ModelFactory

# Set GPU memory growth
set_gpu_memory_growth()

(model_name, appliance) = get_args()

wandb.init(
    project="nilm_multiple_models",
    config=for_model_appliance(model_name, appliance)
)

# Retrieve the configuration from WandB
wandb_config = wandb.config

if wandb_config.continuation:
    try:
        nn_model = tf.keras.models.load_model('../models/{}_model'.format(model_name))
    except Exception as e:
        print("Error loading the model: ", e)
        print("Trying to rebuild the model and load weights...")

        # Rebuild the model
        nn_model = ModelFactory(wandb_config, True).create_model(model_name)

        # Load the weights from the checkpoint files
        nn_model.load_weights('../models/{}_model'.format(model_name))
        print("Model architecture rebuilt and weights loaded successfully!")

        # Compile the model for evaluation
        nn_model.compile(
            metrics=[
                MeanRelativeError(name='MRE'),
                tf.keras.metrics.MeanAbsoluteError(name='MAE')
            ]
        )
else:
    # When it's not a continuation, builds a new model from scratch and load no wights whatsoever
    nn_model = ModelFactory(wandb_config, True).create_model(model_name)

nn_model.summary()

path_to_dataset = '../datasets/ukdale.h5'
print("Fetching data from the dataset located at ", path_to_dataset)
dataset = DataSet(path_to_dataset)

# time series handler for the UK Dale dataset
training_buildings = [1]
timeSeries = TimeSeries(dataset, training_buildings, [2], wandb_config)

m_batch = None
train_gen = timeSeries.getTrainingDataGenerator()
if wandb_config.model in ['bert', 'transformer']:
    X_batch, y_batch, m_batch = train_gen[0]
else:
    X_batch, y_batch = train_gen[0]

# Ensure the shapes match
if wandb_config.model == 'bert':
    print(f"Sample batch shapes - X: {X_batch.shape}, y: {y_batch.shape}, m: {m_batch.shape}")
else:
    print(f"Sample batch shapes - X: {X_batch.shape}, y: {y_batch.shape}")
print("Sample statistics:")
print(f"X mean: {np.mean(X_batch)}, std: {np.std(X_batch)}")
print(f"y mean: {np.mean(y_batch)}, std: {np.std(y_batch)}")
print(f"X range: [{np.min(X_batch)}, {np.max(X_batch)}]")
print(f"y range: [{np.min(y_batch)}, {np.max(y_batch)}]")

print("... The training data is available. Starting training ...")

my_callbacks = [
    # WandbMetricsLogger(log_freq='batch'),
    EarlyStopping(patience=6, monitor='val_MAE', restore_best_weights=True),
    ModelCheckpoint('../models/{}_model'.format(model_name), save_best_only=True, monitor='val_MAE', save_format="tf")
]

# Train the model and track the training process using WandB
history = nn_model.fit(
    timeSeries.getTrainingDataGenerator(),
    epochs=wandb_config.epochs,
    validation_data=timeSeries.getTestDataGenerator(),
    callbacks=my_callbacks
)

# Finish the WandB run
wandb.finish()

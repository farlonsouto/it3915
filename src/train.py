import numpy as np
import wandb
from nilmtk import DataSet
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from cmd_line_input import get_args
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

nn_model = ModelFactory(wandb_config, True).create_model(model_name)

nn_model.summary()

path_to_dataset = '../datasets/ukdale.h5'
print("Fetching data from the dataset located at ", path_to_dataset)
dataset = DataSet(path_to_dataset)

# time series handler for the UK Dale dataset
training_buildings = [1, 3, 4, 5]
timeSeries = TimeSeries(dataset, training_buildings, [2], wandb_config)

train_gen = timeSeries.getTrainingDataGenerator()
X_batch, y_batch, m_batch = train_gen[0]

# Ensure the shapes match
print(f"Sample batch shapes - X: {X_batch.shape}, y: {y_batch.shape}")
print("Sample statistics:")
print(f"X mean: {np.mean(X_batch)}, std: {np.std(X_batch)}")
print(f"y mean: {np.mean(y_batch)}, std: {np.std(y_batch)}")
print(f"X range: [{np.min(X_batch)}, {np.max(X_batch)}]")
print(f"y range: [{np.min(y_batch)}, {np.max(y_batch)}]")

print("... The training data is available. Starting training ...")

my_callbacks = [
    # WandbMetricsLogger(log_freq='batch'),
    EarlyStopping(patience=10, monitor='val_MAE', restore_best_weights=True),
    ModelCheckpoint('../models/{}_model'.format(model_name), save_best_only=True, monitor='MAE', save_format="tf")
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

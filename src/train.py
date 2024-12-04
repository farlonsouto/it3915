import numpy as np
import wandb
from nilmtk import DataSet
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from data.timeseries import TimeSeries
from gpu.gpu_memory_allocation import set_gpu_memory_growth
from model_factory import create_model
from wandb_init import config

# Set GPU memory growth
set_gpu_memory_growth()

wandb.init(
    project="nilm_bert_transformer",
    config=config
)

# Retrieve the configuration from WandB
wandb_config = wandb.config

bert_model = create_model(wandb_config)

path_to_dataset = '../datasets/ukdale.h5'
print("Fetching data from the dataset located at ", path_to_dataset)
dataset = DataSet(path_to_dataset)

# time series handler for the UK Dale dataset
training_buildings_kettle = [1, 3, 4, 5]
training_buildings_fridge = [1, 5]
timeSeries = TimeSeries(dataset, training_buildings_fridge, [2], wandb_config)

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
    # WandbMetricsLogger(log_freq='batch'),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
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

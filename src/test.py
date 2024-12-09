import numpy as np
import tensorflow as tf
import wandb
from nilmtk import DataSet

from cmd_line_input import get_appliance_arg
from custom.metric.regression import MeanRelativeError
from data.timeseries import TimeSeries
from gpu.gpu_memory_allocation import set_gpu_memory_growth
from hyper_params import for_appliance
from model.bert4nilm import BERT4NILM
from plotter import plot_comparison

# Set GPU memory growth
set_gpu_memory_growth()

wandb.init(
    project="nilm_bert_transformer",
    config=for_appliance(get_appliance_arg())
)

# Retrieve the configuration from WandB
wandb_config = wandb.config

try:
    bert_model = tf.keras.models.load_model('../models/bert_model')
except Exception as e:
    print("""Error loading the model: """, e)
    print("Trying the rebuild and load weights approach ...")
    # Rebuild the model architecture
    bert_model = BERT4NILM(wandb_config)

    # Build the model with input shape
    bert_model.build((None, wandb_config.window_size, wandb_config.num_features))

    # Load the weights from the checkpoint files
    bert_model.load_weights('../models/bert_model')
    print("Model architecture rebuilt and weights loaded successfully!")

    # Compile the model for evaluation
    bert_model.compile(
        metrics=[
            MeanRelativeError(name='MRE'),
            tf.keras.metrics.MeanAbsoluteError(name='MAE')
        ]
    )

bert_model.summary()

print("Model loaded successfully!")

# Load the dataset
dataset = DataSet('../datasets/ukdale.h5')

# Prepare the test data generator
timeSeries = TimeSeries(dataset, [2], [2], wandb_config)

test_gen = timeSeries.getTestDataGenerator()

# Evaluate the model on the test data
results = bert_model.evaluate(test_gen)
print("\nModel performance on test data:")
for metric_name, result in zip(bert_model.metrics_names, results):
    print(f"{metric_name}: {result}")

# Get predictions on the test data
X_test, y_test = next(iter(test_gen))  # Get the first batch of test data
predictions = bert_model.predict(X_test)

# Print example predictions
print("\nExample predictions:")
for i in range(5):  # Print the first 5 samples
    print(f"Input: {X_test[i].flatten()}")
    print(f"True appliance power: {y_test[i].flatten()}")
    print(f"Predicted appliance power: {predictions[i].flatten()}")
    print("----")

# Plot the comparison
plot_comparison(test_gen, bert_model)

# Print some statistics
print("\nStatistics:")
print(f"Ground Truth - Min: {np.min(y_test):.2f}, Max: {np.max(y_test):.2f}, Mean: {np.mean(y_test):.2f}")
print(f"Predicted - Min: {np.min(predictions):.2f}, Max: {np.max(predictions):.2f}, Mean: {np.mean(predictions):.2f}")
print(f"Mean Absolute Error: {np.mean(np.abs(y_test - predictions)):.2f}")

# Finish the WandB run
wandb.finish()

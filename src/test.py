import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from nilmtk import DataSet

from custom.metric.classification import Accuracy, F1Score
from custom.metric.regression import MeanRelativeError
from data.timeseries import TimeSeries
from gpu.gpu_memory_allocation import set_gpu_memory_growth
from model.bert4nilm import BERT4NILM
from wandb_init import config

# Set GPU memory growth
set_gpu_memory_growth()

wandb.init(
    project="nilm_bert_transformer",
    config=config
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
    bert_model.build((None, wandb_config.window_size, 1))

    # Load the weights from the checkpoint files
    bert_model.load_weights('../models/bert_model')
    print("Model architecture rebuilt and weights loaded successfully!")

    # Compile the model for evaluation
    bert_model.compile(
        metrics=[
            Accuracy(wandb_config.on_threshold),
            tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            MeanRelativeError(name='MRE'),
            F1Score(on_threshold=wandb_config.on_threshold)
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


# Plotting function
def plot_comparison(y_true, y_pred, title='Energy Consumption: Predicted vs Ground Truth'):
    samples = y_true.shape[0] * y_true.shape[1]
    x = np.arange(samples)

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    plt.figure(figsize=(20, 10))
    plt.plot(x, y_true_flat, label='Ground Truth', color='blue', linewidth=2, alpha=0.7)
    plt.plot(x, y_pred_flat, label='Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)

    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Power (W)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)

    y_min = min(np.min(y_true_flat), np.min(y_pred_flat))
    y_max = max(np.max(y_true_flat), np.max(y_pred_flat))
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()
    plt.show()

    # Plot the difference
    plt.figure(figsize=(20, 10))
    plt.plot(x, y_pred_flat - y_true_flat, label='Prediction Error', color='green', linewidth=2)
    plt.title('Prediction Error', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Error (Predicted - Ground Truth)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Plot the comparison
plot_comparison(y_test, predictions)

# Print some statistics
print("\nStatistics:")
print(f"Ground Truth - Min: {np.min(y_test):.2f}, Max: {np.max(y_test):.2f}, Mean: {np.mean(y_test):.2f}")
print(f"Predicted - Min: {np.min(predictions):.2f}, Max: {np.max(predictions):.2f}, Mean: {np.mean(predictions):.2f}")
print(f"Mean Absolute Error: {np.mean(np.abs(y_test - predictions)):.2f}")

# Finish the WandB run
wandb.finish()

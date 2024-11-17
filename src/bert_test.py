import tensorflow as tf
import wandb
from nilmtk import DataSet

from bert4nilm import BERT4NILM
from bert_wandb_init import config
from custom_metrics import F1Score, Accuracy, MeanRelativeError
from gpu_memory_allocation import set_gpu_memory_growth
from time_series_uk_dale import TimeSeries

# Set GPU memory growth
set_gpu_memory_growth()

wandb.init(
    project="nilm_bert_transformer",
    config=config
)

# Retrieve the configuration from WandB
wandb_config = wandb.config

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
X_test, y_test = test_gen[0]  # Get the first batch of test data
predictions = bert_model.predict(X_test)

# Print example predictions
print("\nExample predictions:")
for i in range(5):  # Print the first 5 samples
    print(f"Input: {X_test[i].flatten()}")
    print(f"True appliance power: {y_test[i].flatten()}")
    print(f"Predicted appliance power: {predictions[i].flatten()}")
    print("----")

# Finish the WandB run
wandb.finish()

import wandb
from nilmtk import DataSet
from tensorflow.keras.models import load_model

from custom_loss import nde_loss
from custom_metrics import mre_metric, f1_score, nde_metric
from time_series_helper import TimeSeriesHelper

# Initialize WandB for tracking
wandb.init(
    project="nilm_bert_transformer_test",
)

# Set up the WandB configuration similar to training
wandb_config = {
    "window_size": 64,
    "batch_size": 128,
    "on_threshold": 2000,
    "appliance": "kettle",
    "model_path": '../models/bert_model.keras',
}

# Load the NILMTK dataset
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-20', end='2015-02-10')  # TODO: Will read from building 5 (five)

# Initialize the TimeSeriesHelper to preprocess the test data
timeSeriesHelper = TimeSeriesHelper(
    dataset,
    window_size=wandb_config['window_size'],
    batch_size=wandb_config['batch_size'],
    appliance=wandb_config['appliance'],
    on_threshold=wandb_config['on_threshold']
)

# Load the test data generator
test_gen = timeSeriesHelper.getTestDataGenerator()

# Load the trained model from disk
model = load_model(wandb_config['model_path'], custom_objects={
    'nde_loss': nde_loss,
    'mre_metric': mre_metric,
    'f1_score': f1_score,
    'nde_metric': nde_metric
})

# Evaluate the model on the test data
results = model.evaluate(test_gen)
print("\nModel performance on test data:")
for metric_name, result in zip(model.metrics_names, results):
    print(f"{metric_name}: {result}")

# Get predictions on the test data
X_test, y_test = test_gen[0]  # Get the first batch of test data
predictions = model.predict(X_test)

# Check that the data is consistently sampled every 6 seconds
print("\nValidating time series consistency:")
test_mains_df = timeSeriesHelper.test_mains
sampling_interval = (test_mains_df.index[1] - test_mains_df.index[0]).total_seconds()
print(f"Sampling interval (seconds): {sampling_interval}")
assert sampling_interval == 6, "Data is not sampled at a 6-second interval!"

# Print example test data and corresponding predictions
print("\nExample predictions:")
for i in range(5):  # Print the first 5 samples
    print(f"Input: {X_test[i].flatten()}")
    print(f"True appliance power: {y_test[i].flatten()}")
    print(f"Predicted appliance power: {predictions[i].flatten()}")
    print("----")

# Finish the WandB run
wandb.finish()

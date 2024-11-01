import tensorflow as tf
import wandb
from nilmtk import DataSet

from bert4nilm import BERT4NILM
from bert_wandb_init import wandb_config
from custom_loss import nde_loss
from custom_metrics import mre_metric, f1_score, nde_metric
from gpu_memory_allocation import set_gpu_memory_growth
from time_series_ampds2 import TimeSeries

set_gpu_memory_growth()

# Initialize WandB for tracking
wandb.init(
    project="nilm_bert_transformer_test",
)

# Rebuild the model architecture
bert_model = BERT4NILM(wandb_config)

# Build the model with input shape
bert_model.build((None, wandb_config.window_size, 1))

# Load the saved weights from the .keras file
bert_model.load_weights('../models/bert_model.keras')
print("Model rebuilt and weights loaded successfully!")

optimizer = tf.keras.optimizers.Adam(
    learning_rate=wandb_config.learning_rate,
    clipnorm=1.0,  # gradient clipping
    clipvalue=0.5
)

# Mapping the loss function from WandB configuration to TensorFlow's predefined loss functions
loss_fn_mapping = {
    "mse": tf.keras.losses.MeanSquaredError(),
    "mae": tf.keras.losses.MeanAbsoluteError(),
    "nde_loss": nde_loss,  # Example of an additional loss function
}

# Get the loss function from the WandB config
loss_fn = loss_fn_mapping.get(wandb_config.loss, tf.keras.losses.MeanSquaredError())  # Default to MSE

# Use bert4nilm_loss from bert_loss.py, and pass any required arguments from wandb_config
# Compile the model
bert_model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.MeanSquaredError(name='mse'),
        mre_metric,
        f1_score,
        nde_metric
    ]
)

bert_model.summary()

# Load the dataset
dataset = DataSet('../datasets/ukdale.h5')

# time series handler for the UK Dale dataset
# timeSeries = TimeSeries(dataset, [1], [1],
#                        wandb_config.window_size, wandb_config.batch_size,
#                        appliance=wandb_config.appliance,
#                        on_threshold=wandb_config.on_threshold)

# time series handler for the AMPds2dataset
timeSeries = TimeSeries(dataset, wandb_config.window_size, wandb_config.batch_size,
                        appliance=wandb_config.appliance)

# Load the test data generator
test_gen = timeSeries.getTestDataGenerator()

# Evaluate the model on the test data
results = bert_model.evaluate(test_gen)
print("\nModel performance on test data:")
for metric_name, result in zip(bert_model.metrics_names, results):
    print(f"{metric_name}: {result}")

# Get predictions on the test data
X_test, y_test = test_gen[0]  # Get the first batch of test data
predictions = bert_model.predict(X_test)

# Print example test data and corresponding predictions
print("\nExample predictions:")
for i in range(5):  # Print the first 5 samples
    print(f"Input: {X_test[i].flatten()}")
    print(f"True appliance power: {y_test[i].flatten()}")
    print(f"Predicted appliance power: {predictions[i].flatten()}")
    print("----")

# Finish the WandB run
wandb.finish()

import tensorflow as tf
from nilmtk import DataSet
from wandb.integration.keras import WandbCallback
import wandb
from soft_dtw_loss_wrapper import DynamicTimeWarping
from time_series_helper import TimeSeriesHelper
from bert4nilm import BERT4NILM  # Assuming your BERT4NILM class is in this file


# Custom metric for Summed Absolute Error (SAE)
def sae_metric(y_ground_truth, y_prediction):
    y_ground_truth = tf.reshape(y_ground_truth, [-1])
    y_prediction = tf.reshape(y_prediction, [-1])
    return tf.abs(tf.reduce_sum(y_ground_truth) - tf.reduce_sum(y_prediction))


# Initialize WandB for tracking
wandb.init(
    project="nilm_bert_transformer",
    config={
        "window_size": 128,
        "batch_size": 512,
        "head_size": 256,  # Hidden size for BERT
        "num_heads": 2,  # Number of attention heads
        "n_layers": 2,  # Number of transformer layers
        "dropout": 0.3,
        "learning_rate": 0.001,
        "epochs": 10,
        "optimizer": "adam",
        "loss": "mean_absolute_error",
        "output_size": 1,  # Assuming output size is 1 for NILM task
    }
)

# Retrieve the configuration from WandB
config = wandb.config

# Load the NILMTK dataset
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-01', end='2015-02-15')

# Helper to preprocess time series data
timeSeriesHelper = TimeSeriesHelper(dataset, config.window_size, config.batch_size)


# Define BERT4NILM args class (assuming you pass the config attributes to your BERT4NILM model)
class BERT4NILMArgs:
    def __init__(self, config):
        self.window_size = config.window_size
        self.drop_out = config.dropout
        self.output_size = config.output_size


bert_args = BERT4NILMArgs(config)

# Instantiate the BERT4NILM model
bert_model = BERT4NILM(bert_args)

# Compile the model using the WandB configurations
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
loss = DynamicTimeWarping(gamma=1.0) if config.loss == "dynamic_time_warping" else config.loss

bert_model.compile(optimizer=optimizer, loss=loss, metrics=['mae', sae_metric])

# Print the model summary
bert_model.summary()

# Train the model and track the training process using WandB
history = bert_model.fit(
    timeSeriesHelper.getTrainingDataGenerator(),
    epochs=config.epochs,
    validation_data=timeSeriesHelper.getTestDataGenerator(),
    callbacks=[WandbCallback()]
)

# Finish the WandB run
wandb.finish()

import tensorflow as tf
import wandb
from nilmtk import DataSet

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

try:
    nn_model = tf.keras.models.load_model('../models/{}_model'.format(model_name))
except Exception as e:
    print("Error loading the model: ", e)
    print("Trying to rebuild the model and load weights...")

    # Rebuild the model
    nn_model = ModelFactory(wandb_config, False).create_model(model_name)

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

nn_model.summary()

print("Model loaded successfully!")

# Load the dataset
dataset = DataSet('../datasets/ukdale.h5')
# time series handler for the UK Dale dataset
test_data = TimeSeries(dataset, [2], [2], wandb_config)

test_gen = test_data.getTestDataGenerator()

nn_model.evaluate(test_gen)

# Finish the WandB run
wandb.finish()

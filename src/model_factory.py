import tensorflow as tf

from custom.loss.bert4nilm import LossFunction
from custom.metric.regression import MeanRelativeError
from model.bert4nilm import BERT4NILM


def create_model(wandb_config):
    # Compile the model using the WandB configurations and the custom loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=wandb_config.learning_rate,
        clipnorm=1.0,  # gradient clipping
        clipvalue=0.5
    )

    # Mapping the loss function from WandB configuration to TensorFlow's predefined loss functions
    loss_fn_mapping = {
        "mse": tf.keras.losses.MeanSquaredError(),
        "mae": tf.keras.losses.MeanAbsoluteError(),
        "bert4nilm_loss": LossFunction(wandb_config)
    }

    # Get the loss function from the WandB config
    loss_fn = loss_fn_mapping.get(wandb_config.loss, tf.keras.losses.MeanSquaredError())  # Default to MSE

    # Instantiate the BERT4NILM model
    model = BERT4NILM(wandb_config)

    # Build the model by providing an input shape
    # NOTICE: The complete input shape is (Batch size, window size, features) where:
    # `None` stands for a flexible, variable batch size.
    # 'window_size` is the number of time steps in the sliding window
    # `1` corresponds the number of features (for now, only one: the power consumption)
    model.build((None, wandb_config.window_size, 5))

    # Use bert4nilm_loss from bert_loss.py, and pass any required arguments from wandb_config
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            MeanRelativeError(name='MRE'),
        ]
    )

    return model

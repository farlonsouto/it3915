import tensorflow as tf

from custom.loss.bert4nilm import LossFunction
from custom.metric.regression import MeanRelativeError
from model.bert4nilm import BERT4NILM
from model.seq2seq import Seq2SeqNILM


class Create:

    def __init__(self, wandb_config):
        self.wandb_config = wandb_config

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=wandb_config.learning_rate,
            clipnorm=1.0,  # gradient clipping
            clipvalue=0.5
        )

        # Mapping the loss function from WandB configuration to TensorFlow's predefined loss functions
        self.loss_fn_mapping = {
            "mse": tf.keras.losses.MeanSquaredError(),
            "mae": tf.keras.losses.MeanAbsoluteError(),
            "bert4nilm_loss": LossFunction(wandb_config)
        }

    def bert4nilm(self):
        # Instantiate the BERT4NILM model
        model = BERT4NILM(self.wandb_config)
        return self.__build_compile(model)

    def seq2seq(self):
        model = Seq2SeqNILM(self.wandb_config)
        return self.__build_compile(model)

    def __build_compile(self, model):
        # Build the model by providing an input shape
        # NOTICE: The complete input shape is (Batch size, window size, features) where:
        # `None` stands for a flexible, variable batch size.
        # 'window_size` is the number of time steps in the sliding window
        # `1` corresponds the number of features (for now, only one: the power consumption)
        model.build((None, self.wandb_config.window_size, self.wandb_config.num_features))

        # Get the loss function from the WandB config
        loss_fn = self.loss_fn_mapping.get(self.wandb_config.loss, tf.keras.losses.MeanSquaredError())  # Default to MSE

        # Use bert4nilm_loss from bert_loss.py, and pass any required arguments from wandb_config
        model.compile(
            optimizer=self.optimizer,
            loss=loss_fn,
            metrics=[
                MeanRelativeError(name='MRE'),
                tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            ]
        )

        return model

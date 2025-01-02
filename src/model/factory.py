import tensorflow as tf
from custom.loss.bert4nilm import LossFunction
from custom.metric.regression import MeanRelativeError

from .bert4nilm import BERT4NILM
from .seq2p import Seq2PointNILM
from .seq2seq import Seq2SeqNILM


class ModelFactory:

    def __init__(self, wandb_config, is_training):
        self.wandb_config = wandb_config
        self.is_training = is_training

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

    def create_model(self, model_name):
        if model_name == "bert":
            return self.__bert4nilm()
        elif model_name == "seq2seq":
            return self.__seq2seq()
        elif model_name == "seq2p":
            return self.__seq2p()
        else:
            raise Exception("Invalid model name:", model_name)

    def __bert4nilm(self):
        # Instantiate the BERT4NILM model
        model = BERT4NILM(self.wandb_config, self.is_training)
        return self.__build_compile(model)

    def __seq2seq(self):
        model = Seq2SeqNILM(self.wandb_config)
        return self.__build_compile(model)

    def __seq2p(self):
        model = Seq2PointNILM(self.wandb_config)
        return self.__build_compile(model)

    def __build_compile(self, model):
        # Build the model by providing an input shape
        # NOTICE: The complete input shape is (Batch size, window size, features) where:
        # `None` stands for a flexible, variable batch size.
        # 'window_size` is the number of time steps in the sliding window
        # `1` corresponds the number of features (for now, only one: the power consumption)
        model.build((None, self.wandb_config.window_size, self.wandb_config.num_features))

        # Get the loss function from the WandB config
        loss_function = self.loss_fn_mapping.get(self.wandb_config.loss,
                                                 tf.keras.losses.MeanSquaredError())  # Default to MSE

        # Use bert4nilm_loss from bert_loss.py, and pass any required arguments from wandb_config
        model.compile(
            optimizer=self.optimizer,
            loss=loss_function,
            metrics=[MeanRelativeError(), "MAE"],
        )

        return model

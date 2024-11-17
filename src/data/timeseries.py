import math
import warnings

import pandas as pd
from nilmtk import DataSet

from .generator import TimeSeriesDataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)


class TimeSeries:
    """
    Encapsulates UK Dale dataset handling intelligence with a focus on loading the data from a single HDF5 format file.
    """

    def __init__(self, dataset: DataSet, training_buildings: list, test_buildings: list, wandb_config):
        self.training_buildings = training_buildings
        self.test_buildings = test_buildings
        self.dataset = dataset
        self.wandb_config = wandb_config
        self.window_size = wandb_config.window_size
        self.batch_size = wandb_config.batch_size
        self.appliance = wandb_config.appliance
        self.mean_power = None
        self.std_power = None
        self._compute_normalization_params()

    def _compute_normalization_params(self):
        """
        Computes normalization parameters (mean and standard deviation) using the training data.
        """
        all_train_mains_power = []

        for building in self.training_buildings:
            train_elec = self.dataset.buildings[building].elec
            train_mains = train_elec.mains()
            mains_data_frame = train_mains.load()

            for train_mains_df in mains_data_frame:
                mains_power = train_mains_df[('power', 'apparent')]
                all_train_mains_power.append(mains_power)

        if all_train_mains_power:
            combined_mains_power = pd.concat(all_train_mains_power, axis=0)
            self.mean_power = combined_mains_power.mean()
            self.std_power = combined_mains_power.std()
            print(f"Mean power: {self.mean_power}")
            print(f"Std power: {self.std_power}")

            if math.isnan(self.mean_power) or math.isnan(self.std_power):
                raise ValueError("Normalization parameters contain NaN values. Check your data preprocessing steps.")
        else:
            raise ValueError("No training data available for normalization.")

    def getTrainingDataGenerator(self):
        return TimeSeriesDataGenerator(
            self.dataset, self.training_buildings, self.appliance, self.mean_power,
            self.std_power, self.wandb_config, is_training=True
        )

    def getTestDataGenerator(self):
        return TimeSeriesDataGenerator(
            self.dataset, self.test_buildings, self.appliance, self.mean_power,
            self.std_power, self.wandb_config, is_training=False
        )

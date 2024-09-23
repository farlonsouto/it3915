import numpy as np
import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.utils import Sequence


class TimeSeriesHelper:
    def __init__(self, dataset: DataSet, window_size: int, batch_size: int):
        self.dataset = dataset
        self.window_size = window_size
        self.batch_size = batch_size
        self.mean_power = None
        self.std_power = None
        self.train_mains = None
        self.test_mains = None

        self._prepare_data()

    def _prepare_data(self):
        train_mains_list = []
        for building in [1, 2, 3, 4]:
            train_elec = self.dataset.buildings[building].elec
            train_mains = train_elec.mains()
            train_mains_df = next(train_mains.load())

            if isinstance(train_mains_df.columns, pd.MultiIndex):
                train_mains_df.columns = ['_'.join(col).strip() for col in train_mains_df.columns.values]

            power_column = [col for col in train_mains_df.columns if 'power' in col.lower()]
            if power_column:
                train_mains_power = train_mains_df[power_column[0]]
                train_mains_list.append(train_mains_power)
            else:
                print(f"No power-related column found in building {building}, skipping.")

        if not train_mains_list:
            raise ValueError("No valid power data found for any of the buildings in the training set.")

        all_train_mains_power = pd.concat(train_mains_list)
        self.mean_power = all_train_mains_power.mean()
        self.std_power = all_train_mains_power.std()

        self.train_mains = (all_train_mains_power - self.mean_power) / self.std_power

        test_elec = self.dataset.buildings[5].elec
        test_mains = test_elec.mains()
        test_mains_df = next(test_mains.load())

        if isinstance(test_mains_df.columns, pd.MultiIndex):
            test_mains_df.columns = ['_'.join(col).strip() for col in test_mains_df.columns.values]

        power_column = [col for col in test_mains_df.columns if 'power' in col.lower()]
        if power_column:
            test_mains_power = test_mains_df[power_column[0]]
        else:
            raise KeyError(f"No power-related column found in Building 5 mains data.")

        self.test_mains = (test_mains_power - self.mean_power) / self.std_power

        print(f"Training data shape: {self.train_mains.shape}")
        print(f"Test data shape: {self.test_mains.shape}")

    def getTrainingDataGenerator(self):
        return TimeSeriesGenerator(self.train_mains, self.window_size, self.batch_size)

    def getTestDataGenerator(self):
        return TimeSeriesGenerator(self.test_mains, self.window_size, self.batch_size)


class TimeSeriesGenerator(Sequence):
    def __init__(self, time_series, window_size, batch_size):
        self.time_series = time_series.values.reshape(-1, 1)
        self.window_size = window_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.time_series) - window_size)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self.time_series[i:i + self.window_size] for i in batch_indexes])
        y = np.array([self.time_series[i + self.window_size] for i in batch_indexes])
        return X, y.reshape(-1, 1)  # Ensure y is 2D

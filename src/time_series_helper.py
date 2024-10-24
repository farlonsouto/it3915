import numpy as np
import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.utils import Sequence


class TimeSeriesHelper:
    def __init__(self, dataset: DataSet, window_size: int, batch_size: int, appliance: str, on_threshold: float):
        self.dataset = dataset
        self.window_size = window_size
        self.batch_size = batch_size
        self.appliance = appliance
        self.on_threshold = on_threshold
        self.mean_power = None
        self.std_power = None
        self.train_mains = None
        self.train_appliance = None
        self.test_mains = None
        self.test_appliance = None
        self.train_status = None
        self.test_status = None
        self._prepare_data()

    def _prepare_data(self):
        train_mains_list = []
        train_appliance_list = []

        # Process Building 1 (training data)
        building = 1
        train_elec = self.dataset.buildings[building].elec
        train_mains = train_elec.mains()
        train_appliance = train_elec[self.appliance]
        train_mains_df = next(train_mains.load())
        train_appliance_df = next(train_appliance.load())

        print(f"Building {building} mains columns: {train_mains_df.columns}")
        print(f"Building {building} appliance columns: {train_appliance_df.columns}")

        # Extract 'power' 'active' from MultiIndex
        train_mains_power = train_mains_df['power']['active']
        train_appliance_power = train_appliance_df['power']['active']

        train_mains_list.append(train_mains_power)
        train_appliance_list.append(train_appliance_power)

        # Process test data (Building 5)
        test_elec = self.dataset.buildings[5].elec
        test_mains = test_elec.mains()
        test_appliance = test_elec[self.appliance]
        test_mains_df = next(test_mains.load())
        test_appliance_df = next(test_appliance.load())

        print(f"Test building mains columns: {test_mains_df.columns}")
        print(f"Test building appliance columns: {test_appliance_df.columns}")

        # Extract 'power' 'active' from MultiIndex for test data
        test_mains_power = test_mains_df['power']['active']
        test_appliance_power = test_appliance_df['power']['active']

        # Prepare training data
        all_train_mains_power = pd.concat(train_mains_list)
        all_train_appliance_power = pd.concat(train_appliance_list)
        self.mean_power = all_train_mains_power.mean()
        self.std_power = all_train_mains_power.std()

        self.train_mains = (all_train_mains_power - self.mean_power) / self.std_power
        self.train_appliance = all_train_appliance_power

        # Prepare test data
        self.test_mains = (test_mains_power - self.mean_power) / self.std_power
        self.test_appliance = test_appliance_power

        print(f"Training data shape: {self.train_mains.shape}, {self.train_appliance.shape}")
        print(f"Test data shape: {self.test_mains.shape}, {self.test_appliance.shape}")

        # Ensure train_appliance and test_appliance are not None before comparison
        if self.train_appliance is not None:
            self.train_status = (self.train_appliance > self.on_threshold).astype(int)
        else:
            raise ValueError("train_appliance is None. Check if the appliance data is loaded correctly.")

        if self.test_appliance is not None:
            self.test_status = (self.test_appliance > self.on_threshold).astype(int)
        else:
            raise ValueError("test_appliance is None. Check if the appliance data is loaded correctly.")

    def getTrainingDataGenerator(self):
        return TimeSeriesGenerator(self.train_mains, self.train_appliance, self.train_status, self.window_size,
                                   self.batch_size)

    def getTestDataGenerator(self):
        return TimeSeriesGenerator(self.test_mains, self.test_appliance, self.test_status, self.window_size,
                                   self.batch_size)


class TimeSeriesGenerator(Sequence):
    def __init__(self, mains_series, appliance_series, status_series, window_size, batch_size):
        self.mains_series = mains_series.values.reshape(-1, 1)
        self.appliance_series = appliance_series.values.reshape(-1, 1)
        self.status_series = status_series.values.reshape(-1, 1)
        self.window_size = window_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.mains_series) - window_size)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self.mains_series[i:i + self.window_size] for i in batch_indexes])
        y = np.array([self.appliance_series[i + self.window_size] for i in batch_indexes])
        s = np.array([self.status_series[i + self.window_size] for i in batch_indexes])
        return X, (y.reshape(-1, 1), s.reshape(-1, 1))  # Return X, (y, s)

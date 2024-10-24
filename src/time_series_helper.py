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
        self._prepare_data()

    def _prepare_data(self):
        train_mains_list = []
        train_appliance_list = []

        # Process Building 1 (training data)
        building = 1
        train_elec = self.dataset.buildings[building].elec
        train_mains = train_elec.mains()
        train_appliance = train_elec[self.appliance]

        # Load mains and appliance data
        train_mains_df = next(train_mains.load())
        train_appliance_df = next(train_appliance.load())

        print(f"Building {building} mains columns: {train_mains_df.columns}")
        print(f"Building {building} appliance columns: {train_appliance_df.columns}")

        # Ensure index is a DatetimeIndex
        if not isinstance(train_mains_df.index, pd.DatetimeIndex):
            raise ValueError("Mains data index is not a DatetimeIndex. Please check the dataset.")

        # Handle duplicate timestamps by averaging them
        if train_mains_df.index.duplicated().any():
            print("Duplicate timestamps found, averaging duplicate entries.")
            train_mains_df = train_mains_df.groupby(train_mains_df.index).mean()

        # Check if frequency is set
        if train_mains_df.index.freq is None:
            print("Frequency is not set for mains data. Resampling mains to match appliance intervals.")
            # Resample mains data to 6-second intervals (matching appliance data)
            train_mains_power = train_mains_df['power']['active'].resample('6S').mean()
        else:
            # Use 6-second mains data as-is
            train_mains_power = train_mains_df['power']['active']

        # Extract appliance data (already at 6-second resolution)
        train_appliance_power = train_appliance_df['power']['active']

        # Align mains and appliance data by their time indexes
        train_mains_power, train_appliance_power = train_mains_power.align(train_appliance_power, join='inner')

        # Debugging print statements
        print("Aligned training mains data: ")
        print(train_mains_power.head())

        print("Aligned training appliance data: ")
        print(train_appliance_power.head())

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

        # Ensure index is a DatetimeIndex for test data
        if not isinstance(test_mains_df.index, pd.DatetimeIndex):
            raise ValueError("Test mains data index is not a DatetimeIndex. Please check the dataset.")

        # Handle duplicate timestamps in test data
        if test_mains_df.index.duplicated().any():
            print("Duplicate timestamps found in test data, averaging duplicate entries.")
            test_mains_df = test_mains_df.groupby(test_mains_df.index).mean()

        # Check if frequency is set for test data
        if test_mains_df.index.freq is None:
            print("Frequency is not set for test mains data. Resampling mains to match appliance intervals.")
            # Resample mains data to 6-second intervals
            test_mains_power = test_mains_df['power']['active'].resample('6S').mean()
        else:
            test_mains_power = test_mains_df['power']['active']

        # Extract appliance data for the test set
        test_appliance_power = test_appliance_df['power']['active']

        # Align mains and appliance data for the test set by their time indexes
        test_mains_power, test_appliance_power = test_mains_power.align(test_appliance_power, join='inner')

        # Debugging print statements
        print("Aligned test mains data: ")
        print(test_mains_power.head())

        print("Aligned test appliance data: ")
        print(test_appliance_power.head())

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

        # Debugging print statements
        print(f"Final training data shape: {self.train_mains.shape}, {self.train_appliance.shape}")
        print(f"Final test data shape: {self.test_mains.shape}, {self.test_appliance.shape}")

    def getTrainingDataGenerator(self):
        return TimeSeriesGenerator(self.train_mains, self.train_appliance, self.window_size, self.batch_size)

    def getTestDataGenerator(self):
        return TimeSeriesGenerator(self.test_mains, self.test_appliance, self.window_size, self.batch_size)


class TimeSeriesGenerator(Sequence):
    def __init__(self, mains_series, appliance_series, window_size, batch_size):
        self.mains_series = mains_series.values.reshape(-1, 1)
        self.appliance_series = appliance_series.values.reshape(-1, 1)
        self.window_size = window_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.mains_series) - window_size)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self.mains_series[i:i + self.window_size] for i in batch_indexes])
        y = np.array([self.appliance_series[i + self.window_size] for i in batch_indexes])
        return X, y

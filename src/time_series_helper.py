import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class TimeSeriesHelper:
    def __init__(self, dataset: DataSet, window_size: int, batch_size: int):
        self.dataset = dataset
        self.window_size = window_size
        self.batch_size = batch_size
        self.mean_power = None
        self.std_power = None
        self.train_mains_reshaped = None
        self.test_mains_reshaped = None

        # Initialize the training and test data
        self._prepare_data()

    def _prepare_data(self):
        # Load mains data from multiple buildings in the training set (for concatenation)
        train_mains_list = []

        # Concatenate mains readings from buildings 1-4 (you can adjust based on the actual dataset)
        for building in [1, 2, 3, 4]:
            train_elec = self.dataset.buildings[building].elec
            train_mains = train_elec.mains()
            train_mains_df = next(train_mains.load())

            # Flatten MultiIndex columns if present
            if isinstance(train_mains_df.columns, pd.MultiIndex):
                train_mains_df.columns = ['_'.join(col).strip() for col in train_mains_df.columns.values]

            train_mains_power = train_mains_df['power_active']
            train_mains_list.append(train_mains_power)

        # Concatenate the data from all buildings into a single series for normalization
        all_train_mains_power = pd.concat(train_mains_list)

        # Normalize data using the mean and standard deviation of the concatenated data
        self.mean_power = all_train_mains_power.mean()
        self.std_power = all_train_mains_power.std()

        train_mains_normalized = (all_train_mains_power - self.mean_power) / self.std_power

        # Load test data from Building 5
        test_elec = self.dataset.buildings[5].elec
        test_mains = test_elec.mains()
        test_mains_df = next(test_mains.load())

        # Flatten MultiIndex columns if present
        if isinstance(test_mains_df.columns, pd.MultiIndex):
            test_mains_df.columns = ['_'.join(col).strip() for col in test_mains_df.columns.values]

        test_mains_power = test_mains_df['power_active']
        test_mains_normalized = (test_mains_power - self.mean_power) / self.std_power  # Use train set mean and std

        # Reshape the data into the form expected by the model
        self.train_mains_reshaped = train_mains_normalized.values.reshape(-1, 1)
        self.test_mains_reshaped = test_mains_normalized.values.reshape(-1, 1)

    def getTrainingDataGenerator(self):
        # Prepare training data generator using TimeSeriesGenerator
        return TimeseriesGenerator(self.train_mains_reshaped, self.train_mains_reshaped,
                                   length=self.window_size, batch_size=self.batch_size)

    def getTestDataGenerator(self):
        # Prepare test data generator using TimeSeriesGenerator
        return TimeseriesGenerator(self.test_mains_reshaped, self.test_mains_reshaped,
                                   length=self.window_size, batch_size=self.batch_size)

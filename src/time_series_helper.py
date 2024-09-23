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

            # Debug: Print available columns to help understand the structure
            print(f"Available columns for Building {building}: {train_mains_df.columns}")

            # Skip buildings with no available columns
            if train_mains_df.empty:
                print(f"Skipping Building {building} due to lack of data.")
                continue

            # Check if 'power_active' exists, otherwise search for a power-related column
            if 'power_active' in train_mains_df.columns:
                train_mains_power = train_mains_df['power_active']
            else:
                # Fall back: Look for a column that contains 'power'
                power_column = [col for col in train_mains_df.columns if 'power' in col.lower()]
                if power_column:
                    train_mains_power = train_mains_df[power_column[0]]
                else:
                    print(f"No power-related column found in building {building}, skipping.")
                    continue

            train_mains_list.append(train_mains_power)

        if not train_mains_list:
            raise ValueError("No valid power data found for any of the buildings in the training set.")

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

        # Check if 'power_active' exists in the test set, otherwise search for a power-related column
        if 'power_active' in test_mains_df.columns:
            test_mains_power = test_mains_df['power_active']
        else:
            power_column = [col for col in test_mains_df.columns if 'power' in col.lower()]
            if power_column:
                test_mains_power = test_mains_df[power_column[0]]
            else:
                raise KeyError(f"No power-related column found in Building 5 mains data.")

        # Normalize test data using the same mean and std from the training data
        test_mains_normalized = (test_mains_power - self.mean_power) / self.std_power

        # Reshape the data into the form expected by the model
        self.train_mains_reshaped = train_mains_normalized.values.reshape(-1, 1)
        self.test_mains_reshaped = test_mains_normalized.values.reshape(-1, 1)

        print(f"time series helper - Training data shape: {self.train_mains_reshaped.shape}")
        print(f"time series helper - Test data shape: {self.test_mains_reshaped.shape}")

    def getTrainingDataGenerator(self):
        # Prepare training data generator using TimeSeriesGenerator
        return TimeseriesGenerator(self.train_mains_reshaped, self.train_mains_reshaped,
                                   length=self.window_size, batch_size=self.batch_size)

    def getTestDataGenerator(self):
        # Prepare test data generator using TimeSeriesGenerator
        return TimeseriesGenerator(self.test_mains_reshaped, self.test_mains_reshaped,
                                   length=self.window_size, batch_size=self.batch_size)

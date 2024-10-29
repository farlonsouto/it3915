import numpy as np
import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.utils import Sequence


class TimeSeriesHelper:
    """
    Encapsulates UK Dale dataset handling intelligence.

    Data are available for each house as follows:
    House 1: start 09/11/2012, end 26/04/2017
    House 2: start 17/02/2013, end 10/10/2013
    House 3: start 27/02/2013, end 08/04/2013
    House 4: start 09/03/2013, end 01/10/2013
    House 5: start 29/06/2014, end 13/11/2014


    """

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

    def _handle_timeseries(self, mains_df, appliance_df, building_num, is_training=True):
        """Helper method to process individual building data"""
        print(f"Processing {'training' if is_training else 'test'} data from building {building_num}")

        # Extract power readings
        if ('power', 'active') in mains_df.columns:
            mains_power = mains_df[('power', 'active')]
        elif 'power' in mains_df.columns:  # Se "power" estiver no formato de coluna simples
            mains_power = mains_df['power']
        else:
            raise KeyError(
                f"The column 'power' was not found inside mains_df for the building {building_num}. Available columns "
                f"are: {mains_df.columns}")

        appliance_power = appliance_df[('power', 'active')]

        # Handle missing values in mains power
        mains_power = mains_power.fillna(method='ffill').fillna(method='bfill')

        # Handle missing values in appliance power - fill with zeros as appliances are usually off
        appliance_power = appliance_power.fillna(0)

        # Remove duplicates before resampling
        mains_power = mains_power[~mains_power.index.duplicated(keep='first')]
        appliance_power = appliance_power[~appliance_power.index.duplicated(keep='first')]

        # Resample both series to 6S frequency using interpolation for mains
        # and forward fill for appliance (since appliance states tend to persist)
        mains_power = mains_power.resample('6S').interpolate(method='linear')
        appliance_power = appliance_power.resample('6S').ffill()

        # Align the series
        mains_power, appliance_power = mains_power.align(appliance_power, join='inner')

        # Remove any remaining NaN values by dropping those rows
        valid_idx = ~(mains_power.isna() | appliance_power.isna())
        mains_power = mains_power[valid_idx]
        appliance_power = appliance_power[valid_idx]

        print(f"Processed data shape: {mains_power.shape}")
        print(f"Any NaN in mains: {mains_power.isna().any()}")
        print(f"Any NaN in appliance: {appliance_power.isna().any()}")

        return mains_power, appliance_power

    def _prepare_data(self):
        """Prepare the training and testing data"""
        train_mains_list = []
        train_appliance_list = []

        try:
            # Buildings 1, 3, 4, 5 for training
            train_buildings = [1, 3, 4, 5]
            for building in train_buildings:
                train_elec = self.dataset.buildings[building].elec
                train_mains = train_elec.mains()
                train_appliance = train_elec[self.appliance]

                train_mains_df = next(train_mains.load())
                train_appliance_df = next(train_appliance.load())

                mains_power, appliance_power = self._handle_timeseries(
                    train_mains_df,
                    train_appliance_df,
                    building,
                    is_training=True
                )

                train_mains_list.append(mains_power)
                train_appliance_list.append(appliance_power)

            # Building 2 for testing
            test_building = 2
            test_elec = self.dataset.buildings[test_building].elec
            test_mains = test_elec.mains()
            test_appliance = test_elec[self.appliance]

            test_mains_df = next(test_mains.load())
            test_appliance_df = next(test_appliance.load())

            test_mains_power, test_appliance_power = self._handle_timeseries(
                test_mains_df,
                test_appliance_df,
                test_building,
                is_training=False
            )

            # Combine all training data
            all_train_mains_power = pd.concat(train_mains_list)
            all_train_appliance_power = pd.concat(train_appliance_list)

            # Calculate normalization parameters from training data
            self.mean_power = all_train_mains_power.mean()
            print(f"Mean power: {self.mean_power}")
            self.std_power = all_train_mains_power.std()
            print(f"Std power: {self.std_power}")

            # Normalize the data
            self.train_mains = (all_train_mains_power - self.mean_power) / self.std_power
            self.train_appliance = all_train_appliance_power

            self.test_mains = (test_mains_power - self.mean_power) / self.std_power
            self.test_appliance = test_appliance_power

            # Final validation
            if np.any(np.isnan(self.train_mains)) or np.any(np.isnan(self.train_appliance)):
                raise ValueError("Training data still contains NaN values after processing")

            if np.any(np.isnan(self.test_mains)) or np.any(np.isnan(self.test_appliance)):
                raise ValueError("Test data still contains NaN values after processing")

            # Print final shapes and statistics
            print("\nFinal data statistics:")
            print(f"Training samples: {len(self.train_mains)}")
            print(f"Test samples: {len(self.test_mains)}")
            print(f"Training mains range: [{self.train_mains.min():.2f}, {self.train_mains.max():.2f}]")
            print(f"Training appliance range: [{self.train_appliance.min():.2f}, {self.train_appliance.max():.2f}]")

        except Exception as e:
            print(f"Error during data preparation: {str(e)}")
            raise

    def getTrainingDataGenerator(self):
        return TimeSeriesGenerator(self.train_mains, self.train_appliance,
                                   self.window_size, self.batch_size)

    def getTestDataGenerator(self):
        return TimeSeriesGenerator(self.test_mains, self.test_appliance,
                                   self.window_size, self.batch_size)


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
        y = np.array([self.appliance_series[i:i + self.window_size] for i in batch_indexes])
        # Ensure y has the shape (batch_size, window_size, 1)
        y = y.reshape(self.batch_size, self.window_size, 1)
        return X, y

import numpy as np
import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.utils import Sequence


class TimeSeries:
    """
    Encapsulates AMPds2 dataset handling intelligence with focus on loading the data from a single HDF5 format file.
    """

    def __init__(self, dataset: DataSet, window_size: int, batch_size: int, appliance: str,
                 train_fraction: float = 0.8):
        self.dataset = dataset
        self.window_size = window_size
        self.batch_size = batch_size
        self.appliance = appliance
        self.train_fraction = train_fraction
        self.mean_power = None
        self.std_power = None
        self.train_data = None
        self.test_data = None
        self._load_and_split_data()
        self._compute_normalization_params()

    def _load_and_split_data(self):
        building = self.dataset.buildings[1]  # AMPds2 has only one building
        mains = building.elec.mains()
        appliance_meter = building.elec[self.appliance]

        # Load mains and appliance data
        mains_data = next(mains.load())
        appliance_data = next(appliance_meter.load())

        # Ensure both DataFrames have the same index
        common_index = mains_data.index.intersection(appliance_data.index)
        mains_data = mains_data.loc[common_index]
        appliance_data = appliance_data.loc[common_index]

        # Active Power is the actual power which is really transferred to the load such as transformer, induction motors,
        # generators etc. and dissipated in the circuit.  Denoted by (P) and measured in units of Watts (W) i.e. The unit
        # of Real or Active power is Watt where 1W = 1V x 1 A.

        # Extract 'power' column if it exists, otherwise use the first column
        mains_power = mains_data['power']['active']
        appliance_power = appliance_data['power']['active']

        # Rename columns
        mains_power = mains_power.rename('mains')
        appliance_power = appliance_power.rename('appliance')

        # Combine mains and appliance data
        combined_data = pd.concat([mains_power, appliance_power], axis=1)

        # Ensure the index is sorted
        combined_data = combined_data.sort_index()

        # Split data
        split_point = int(len(combined_data) * self.train_fraction)
        self.train_data = combined_data.iloc[:split_point]
        self.test_data = combined_data.iloc[split_point:]

    def _compute_normalization_params(self):
        self.mean_power = self.train_data['mains'].mean()
        self.std_power = self.train_data['mains'].std()
        print(f"Mean power: {self.mean_power}")
        print(f"Std power: {self.std_power}")

    def getTrainingDataGenerator(self):
        return TimeSeriesDataGenerator(self.train_data, self.mean_power, self.std_power,
                                       self.window_size, self.batch_size)

    def getTestDataGenerator(self):
        return TimeSeriesDataGenerator(self.test_data, self.mean_power, self.std_power,
                                       self.window_size, self.batch_size)


class TimeSeriesDataGenerator(Sequence):
    def __init__(self, data, mean_power, std_power, window_size, batch_size):
        self.data = data
        self.mean_power = mean_power
        self.std_power = std_power
        self.window_size = window_size
        self.batch_size = batch_size
        self.total_samples = len(self.data) - self.window_size + 1

    def __len__(self):
        return self.total_samples // self.batch_size

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        batch_X, batch_y = [], []
        for i in range(start_idx, end_idx):
            if i + self.window_size > len(self.data):
                i = len(self.data) - self.window_size

            mains_window = self.data['mains'].iloc[i:i + self.window_size].values
            appliance_window = self.data['appliance'].iloc[i:i + self.window_size].values

            mains_window = (mains_window - self.mean_power) / (self.std_power + 1e-8)

            batch_X.append(mains_window)
            batch_y.append(appliance_window)

        return np.array(batch_X), np.array(batch_y)


# Example usage
if __name__ == "__main__":
    dataset = DataSet('../datasets/AMPds2.h5')
    window_size = 64  # 1 hour (assuming 1-minute intervals)
    batch_size = 64
    appliance = 'fridge'  # 'FRE' is NOT the identifier for the fridge in AMPds2

    time_series = TimeSeries(dataset, window_size, batch_size, appliance)
    train_gen = time_series.getTrainingDataGenerator()
    test_gen = time_series.getTestDataGenerator()

    # Example: Get the first batch from the training generator
    X_batch, y_batch = train_gen[0]
    print(f"Batch shape: X={X_batch.shape}, y={y_batch.shape}")

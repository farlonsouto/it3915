import numpy as np
import pandas as pd
from nilmtk import DataSet
from sklearn.model_selection import KFold
from tensorflow.keras.utils import Sequence


class TimeSeries:
    """
    Encapsulates AMPds2 dataset handling intelligence with focus on loading the data from a single HDF5 format file.
    Supports k-fold cross-validation.
    """

    def __init__(self, dataset: DataSet, window_size: int, batch_size: int, appliance: str, k_folds: int = 5):
        self.dataset = dataset
        self.window_size = window_size
        self.batch_size = batch_size
        self.appliance = appliance
        self.k_folds = k_folds
        self.mean_power = None
        self.std_power = None
        self.combined_data = None
        self.folds = None
        self._load_data()
        self._create_folds()

    def _load_data(self):
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

        # Extract 'power' column if it exists, otherwise use the first column
        mains_power = mains_data['power']['active']
        appliance_power = appliance_data['power']['active']

        # Rename columns
        mains_power = mains_power.rename('mains')
        appliance_power = appliance_power.rename('appliance')

        # Combine mains and appliance data
        self.combined_data = pd.concat([mains_power, appliance_power], axis=1)

        # Ensure the index is sorted
        self.combined_data = self.combined_data.sort_index()

    def _create_folds(self):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        self.folds = list(kf.split(self.combined_data))

    def _compute_normalization_params(self, train_data):
        self.mean_power = train_data['mains'].mean()
        self.std_power = train_data['mains'].std()
        print(f"Mean power: {self.mean_power}")
        print(f"Std power: {self.std_power}")

    def get_fold(self, fold_index):
        if fold_index < 0 or fold_index >= self.k_folds:
            raise ValueError(f"Fold index must be between 0 and {self.k_folds - 1}")

        train_index, test_index = self.folds[fold_index]
        train_data = self.combined_data.iloc[train_index]
        test_data = self.combined_data.iloc[test_index]

        self._compute_normalization_params(train_data)

        train_gen = TimeSeriesDataGenerator(train_data, self.mean_power, self.std_power,
                                            self.window_size, self.batch_size)
        test_gen = TimeSeriesDataGenerator(test_data, self.mean_power, self.std_power,
                                           self.window_size, self.batch_size)

        return train_gen, test_gen


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
    appliance = 'fridge'
    k_folds = 5

    time_series = TimeSeries(dataset, window_size, batch_size, appliance, k_folds)

    for fold in range(k_folds):
        print(f"\nFold {fold + 1}/{k_folds}")
        train_gen, test_gen = time_series.get_fold(fold)

        # Example: Get the first batch from the training generator
        X_batch, y_batch = train_gen[0]
        print(f"Training batch shape: X={X_batch.shape}, y={y_batch.shape}")

        # Example: Get the first batch from the test generator
        X_batch, y_batch = test_gen[0]
        print(f"Test batch shape: X={X_batch.shape}, y={y_batch.shape}")

        # Here you would typically train and evaluate your model for this fold

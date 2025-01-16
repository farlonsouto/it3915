import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


class NILMDataset:
    """Handles data loading and preprocessing for NILM applications."""

    def __init__(self, params):
        """
        Initialize dataset parameters.
        Args:
            params: Dictionary containing 'window_size', 'on_power_threshold',
                   'max_power', 'mean_power', 'std_power'
        """
        self.window_size = params['window_size']
        self.on_threshold = params['on_power_threshold']
        self.max_power = params['max_power']
        self.mean = params['mean_power']
        self.std = params['std_power']

    def load_and_preprocess(self, mains_power, appliance_power):
        """
        Load and preprocess power data.
        Args:
            mains_power: Pandas Series with mains readings
            appliance_power: Pandas Series with appliance readings
        Returns:
            Tuple of aligned and preprocessed mains and appliance data
        """
        # Align timestamps
        aligned_mains, aligned_app = self._align_data(mains_power, appliance_power)

        # Preprocess
        mains_proc = self._normalize(aligned_mains)
        app_proc = self._clip_and_normalize(aligned_app)

        return mains_proc, app_proc

    def _align_data(self, mains, appliance):
        """
        Align mains and appliance data on timestamps.
        Args:
            mains: Mains power time series
            appliance: Appliance power time series
        Returns:
            Tuple of aligned mains and appliance series
        """
        mains = mains.sort_index()
        appliance = appliance.sort_index()

        # Remove duplicates
        mains = mains[~mains.index.duplicated()]
        appliance = appliance[~appliance.index.duplicated()]

        # Align series
        return mains.align(appliance, join='inner', method='pad', limit=1)

    def _normalize(self, data):
        """
        Apply z-score normalization.
        Args:
            data: Power readings to normalize
        Returns:
            Normalized data
        """
        return (data - self.mean) / self.std

    def _clip_and_normalize(self, data):
        """
        Clip values to valid range and normalize.
        Args:
            data: Power readings to process
        Returns:
            Processed data
        """
        clipped = np.clip(data, self.on_threshold, self.max_power)
        return self._normalize(clipped)


class Seq2PointGenerator(Sequence):
    """Generates sequence-to-point training data for NILM."""

    def __init__(self, dataset, window_size, batch_size, stride=1):
        """
        Initialize the generator.
        Args:
            dataset: NILMDataset instance
            window_size: Size of input windows
            batch_size: Number of samples per batch
            stride: Window stride for training
        """
        self.dataset = dataset
        self.window_size = window_size
        self.batch_size = batch_size
        self.stride = stride
        self.mains_windows = []
        self.appliance_midpoints = []

    def prepare_windows(self, mains, appliance):
        """
        Prepare sliding windows for training.
        Args:
            mains: Preprocessed mains power data
            appliance: Preprocessed appliance power data
        """
        # Create sliding windows
        for i in range(0, len(mains) - self.window_size + 1, self.stride):
            self.mains_windows.append(mains[i:i + self.window_size])
            midpoint = i + self.window_size // 2
            self.appliance_midpoints.append(appliance[midpoint])

    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.mains_windows) // self.batch_size

    def __getitem__(self, idx):
        """
        Get batch of windows and corresponding midpoint values.
        Args:
            idx: Batch index
        Returns:
            Tuple of (input windows, target midpoint values)
        """
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size

        batch_x = self.mains_windows[start_idx:end_idx]
        batch_y = self.appliance_midpoints[start_idx:end_idx]

        return np.array(batch_x), np.array(batch_y)
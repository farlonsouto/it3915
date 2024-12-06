import numpy as np
import pandas as pd
from nilmtk import TimeFrame
from tensorflow.keras.utils import Sequence


class TimeSeriesDataGenerator(Sequence):
    def __init__(self, dataset, buildings, appliance, mean_power, std_power, wandb_config, is_training=True):
        self.dataset = dataset
        self.buildings = buildings
        self.appliance = appliance
        self.mean_power = mean_power
        self.std_power = std_power
        self.window_size = wandb_config.window_size
        self.batch_size = wandb_config.batch_size
        self.max_power = wandb_config.max_power
        self.on_threshold = wandb_config.on_threshold
        self.min_on_duration = wandb_config.min_on_duration
        self.min_off_duration = wandb_config.min_off_duration
        self.is_training = is_training
        self.data_generator = self._data_generator()
        self.total_samples = self._count_samples()

    def _data_generator(self):
        """
        Simplified generator to load mains and appliance data in aligned chunks.
        Divides the data into sub-intervals and processes each interval sequentially.
        """

        for building in self.buildings:
            elec = self.dataset.buildings[building].elec
            mains = elec.mains()
            appliance = elec[self.appliance]

            # Explicitly considering  only the overlapping period
            start_date = max(appliance.get_timeframe().start, mains.get_timeframe().start)
            end_date = min(appliance.get_timeframe().end, mains.get_timeframe().end)

            current_start = start_date
            current_end = start_date
            while current_end < end_date:

                current_end = current_start + self._time_delta()
                if current_end >= end_date:
                    current_end = end_date

                time_frame_of_interest = TimeFrame(start=current_start, end=current_end)

                # Load data for the current interval
                appliance_generator = appliance.power_series(
                    ac_type='active', sections=[time_frame_of_interest],
                )
                mains_generator = mains.power_series(
                    ac_type='apparent', sections=[time_frame_of_interest],
                )

                mains_power = next(mains_generator)
                appliance_power = next(appliance_generator)

                # Performs the pre-processing of the data:
                mains_power, appliance_power = self._process_data(mains_power, appliance_power)

                stride = self.window_size
                if self.is_training:
                    stride = 1
                # Yield the data in the expected window size
                for i in range(0, len(mains_power) - self.window_size + 1, stride):
                    yield mains_power[i:i + self.window_size], appliance_power[i:i + self.window_size]

    def _time_delta(self):
        """ The time delta in seconds. For convenience, a multiple of the self.window_size to enable dividing the
        chunks in regular window sizes. Multiplied by 6 because the sampling rate is 6 seconds. """
        return pd.Timedelta(self.window_size * 6 * 100, 'seconds')

    def _process_data(self, mains_power, appliance_power):
        """
        Processes mains and appliance power data, aligns them with tolerance, and applies transformations.

        Parameters:
        - mains_power: Panda Series populated with the mains power data
        - appliance_power: Panda Series populated with the appliance power data

        Returns:
        - Tuple of processed mains and appliance power as numpy arrays
        """

        # Sort indices to avoid performance warnings
        mains_power = mains_power.sort_index()
        appliance_power = appliance_power.sort_index()

        # Remove any possible duplicated indices
        mains_power = mains_power[~mains_power.index.duplicated(keep='first')]
        appliance_power = appliance_power[~appliance_power.index.duplicated(keep='first')]

        # Align the series
        mains_power, appliance_power = mains_power.align(appliance_power, join='inner', method='pad', limit=1)

        # Convert appliance power to binary states
        appliance_status = appliance_power > self.on_threshold
        appliance_status = self._apply_min_durations(
            appliance_status, self.min_on_duration, self.min_off_duration
        )
        appliance_power = appliance_power * appliance_status

        # restores the ones, which are the original values
        appliance_power = appliance_power.replace(0.0, 1.0)

        # Mask for values to ignore
        mask = appliance_power == 1.0
        # Apply clipping only where the mask is False
        appliance_power = appliance_power.where(mask,
                                                appliance_power.clip(lower=self.on_threshold, upper=self.max_power))

        # Normalize mains power
        # mains_power = (mains_power - self.mean_power) / (self.std_power + 1e-8)

        # TODO: With reshaping, the timestamp is NOT one of the features. Should it be?
        # Convert to numpy arrays
        mains_power = mains_power.values.reshape(-1, 1)
        appliance_power = appliance_power.values.reshape(-1, 1)

        return mains_power, appliance_power

    def _apply_min_durations(self, appliance_status, min_on_duration, min_off_duration):
        """
        Apply minimum on/off durations to appliance status.
        """
        appliance_status = appliance_status.astype(int)
        status_changes = appliance_status.diff().fillna(0)

        # Enforce minimum on durations
        on_groups = (status_changes == 1).cumsum() * appliance_status
        on_durations = on_groups.groupby(on_groups).transform('count') * 6
        appliance_status[on_durations < min_on_duration] = 0

        # Enforce minimum off durations
        off_groups = (status_changes == -1).cumsum() * (1 - appliance_status)
        off_durations = off_groups.groupby(off_groups).transform('count') * 6
        appliance_status[off_durations < min_off_duration] = 1

        return appliance_status

    def _count_samples(self):
        """
        Count the total number of samples across all buildings and chunks.
        The count is adjusted for the window size.
        """
        total_samples = 0
        for building in self.buildings:
            elec = self.dataset.buildings[building].elec
            mains = elec.mains()
            appliance = elec[self.appliance]

            # Overlapping timeframe
            start_date = max(appliance.get_timeframe().start, mains.get_timeframe().start)
            end_date = min(appliance.get_timeframe().end, mains.get_timeframe().end)

            # Total samples for the overlapping timeframe
            current_start = start_date
            current_end = start_date

            while current_end < end_date:
                current_end = current_start + self._time_delta()
                if current_end >= end_date:
                    current_end = end_date

                # Timeframe of interest
                time_frame_of_interest = TimeFrame(start=current_start, end=current_end)

                # Load appliance data to estimate samples
                appliance_generator = appliance.power_series(
                    ac_type="active", sections=[time_frame_of_interest]
                )
                try:
                    appliance_chunk = next(appliance_generator)
                    total_samples += len(appliance_chunk)
                except StopIteration:
                    continue

                # Move to the next time chunk
                current_start = current_end

        # Adjust for the sliding window size
        return total_samples // self.window_size

    def __len__(self):
        """
        Calculate the number of batches in one epoch.
        """
        return self.total_samples // self.batch_size

    def __getitem__(self, index):
        """
        Fetches a batch of data using the generator.
        """
        batch_X, batch_y = [], []
        for _ in range(self.batch_size):
            try:
                X, y = next(self.data_generator)
                batch_X.append(X)
                batch_y.append(y)
            except StopIteration:
                # Reset the generator and fetch the next batch
                self.data_generator = self._data_generator()
                X, y = next(self.data_generator)
                batch_X.append(X)
                batch_y.append(y)

        return np.array(batch_X), np.array(batch_y)

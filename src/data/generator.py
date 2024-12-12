import numpy as np
import pandas as pd
from data.adjustment import Augment, Balance
from nilmtk import TimeFrame
from tensorflow.keras.utils import Sequence


class TimeSeriesDataGenerator(Sequence):
    """ Generates
        a) An x (aggregated reading, input) with 1 (one) single channels:
            1. Mains power (mixed AC types)
        b) An y (appliance reading, ground truth) with 1 channel:
            1. Appliance power (always wit the AC type active)
        At the end, yields multiple pairs (x, y) where x and y are aligned according to their original timestamps
        and have shapes (window size, 5) and (window size, 1) respectively.
    """

    def __init__(self, dataset, buildings, appliance, normalization_params, wandb_config, is_training=True):
        self.dataset = dataset
        self.buildings = buildings
        self.appliance = appliance
        self.normalization_params = normalization_params
        self.window_size = wandb_config.window_size
        self.batch_size = wandb_config.batch_size
        self.max_power = wandb_config.max_power
        self.on_threshold = wandb_config.on_threshold
        self.min_on_duration = wandb_config.min_on_duration
        self.window_stride = wandb_config.window_stride
        self.wandb_config = wandb_config
        self.balance_enabled = wandb_config.balance_enabled
        self.add_artificial_activations = wandb_config.add_artificial_activations
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
            aggregated = elec.mains()

            appliance = None
            try:
                appliance = elec[self.appliance]
            except KeyError:
                print(f"Appliance {self.appliance} not available in the Building {building}")
                continue

            # Explicitly considering  only the overlapping period
            start_date = max(appliance.get_timeframe().start, aggregated.get_timeframe().start)
            end_date = min(appliance.get_timeframe().end, aggregated.get_timeframe().end)

            current_start = start_date
            current_end = start_date
            while current_end < end_date:

                current_end = current_start + self._time_delta()
                if current_end >= end_date:
                    current_end = end_date

                time_frame_of_interest = TimeFrame(start=current_start, end=current_end)

                # Load data for the current interval

                ac_type_aggregated = 'apparent'
                # UK Dale's buildings always presents apparent, but this impl gives priority to the active power
                if 'active' in aggregated.available_ac_types('power'):
                    ac_type_aggregated = 'active'

                aggregated_generator = aggregated.power_series(
                    ac_type=ac_type_aggregated, sections=[time_frame_of_interest],
                )

                appliance_generator = appliance.power_series(
                    ac_type='active', sections=[time_frame_of_interest],
                )

                mains_power = next(aggregated_generator)
                appliance_power = next(appliance_generator)

                # Performs the pre-processing of the data:
                mains_power, appliance_power = self._process_data(mains_power, appliance_power, ac_type_aggregated)

                stride = self.window_size
                if self.is_training:
                    stride = self.window_stride
                # Yield the data in the expected window size
                for i in range(0, len(mains_power) - self.window_size + 1, stride):
                    yield mains_power[i:i + self.window_size], appliance_power[i:i + self.window_size]

    def _time_delta(self):
        """ The time delta in seconds. For convenience, a multiple of the self.window_size to enable dividing the
        chunks in regular window sizes. Multiplied by 6 because the sampling rate is 6 seconds.
         """
        return pd.Timedelta(self.window_size * 6 * 100, 'seconds')

    def _process_data(self, aggregated, appliance_power, ac_type_aggregated):
        """
        Processes mains and appliance power data, aligns them with tolerance, and applies transformations.

        Parameters:
        - mains_power: Panda Series populated with the mains power data
        - appliance_power: Panda Series populated with the appliance power data

        Returns:
        - Tuple of processed mains and appliance power as numpy arrays
        """

        # Sort indices to avoid performance warnings
        aggregated = aggregated.sort_index()
        appliance_power = appliance_power.sort_index()

        # Remove any possible duplicated indices
        aggregated = aggregated[~aggregated.index.duplicated(keep='first')]
        appliance_power = appliance_power[~appliance_power.index.duplicated(keep='first')]

        # Align the series
        aggregated, appliance_power = aggregated.align(appliance_power, join='inner', method='pad', limit=1)

        # Mask for values to ignore
        mask = appliance_power == 1.0
        # Apply clipping only where the mask is False
        appliance_power = appliance_power.where(mask,
                                                appliance_power.clip(lower=self.on_threshold, upper=self.max_power))

        if self.appliance in ['kettle', 'microwave', 'dish washer', 'washer']:
            # Handling the least used appliances: rarely ON and, when ON, for short periods.
            augment = Augment(self.wandb_config, self.normalization_params)
            balance = Balance(self.wandb_config, self.normalization_params)
            if self.add_artificial_activations:
                aggregated, appliance_power = augment.with_artificial_activations(aggregated, appliance_power)
            if self.balance_enabled:
                aggregated, appliance_power = balance.on_off_periods(aggregated, appliance_power)

            # Drops the DatetimeIndex: The indexed series turn into a numpy array
        appliance_power = appliance_power.values.reshape(-1, 1)
        aggregated = aggregated.values.reshape(-1, 1)

        return aggregated, appliance_power

    def _count_samples(self):
        """
        Count the total number of samples across all buildings and chunks.
        The count is adjusted for the window size.
        """
        total_samples = 0
        for building in self.buildings:
            elec = self.dataset.buildings[building].elec
            mains = elec.mains()
            appliance = None
            try:
                appliance = elec[self.appliance]
            except KeyError:
                continue

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
        Adjusted to work for Seq2Point by using midpoint targets.
        """
        batch_X, batch_y = [], []
        for _ in range(self.batch_size):
            try:
                X, y = next(self.data_generator)
                # For Seq2Point, use the midpoint of y as the target
                if self.wandb_config.model == 'seq2p':
                    midpoint = len(y) // 2
                    y = y[midpoint]
                batch_X.append(X)
                batch_y.append(y)
            except StopIteration:
                # Reset the generator and fetch the next batch
                self.data_generator = self._data_generator()
                X, y = next(self.data_generator)
                if self.wandb_config.model == 'seq2p':
                    midpoint = len(y) // 2
                    y = y[midpoint]
                batch_X.append(X)
                batch_y.append(y)

        # Only reshape batch y for the Seq2Point model
        # if self.wandb_config.model == 'seq2p':
        #   return np.array(batch_X), np.array(batch_y).reshape(-1, 1)
        return np.array(batch_X), np.array(batch_y)

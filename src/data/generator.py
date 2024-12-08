import numpy as np
import pandas as pd
from nilmtk import TimeFrame
from tensorflow.keras.utils import Sequence


class TimeSeriesDataGenerator(Sequence):
    """ Generates
        a) An X (aggregated reading, input) with 5 (five) channels:
            1. Mains power
            2. Hour of the day (0-23)
            3. Minute (0 - 59)
            4. Second (0 - 59)
            5. AC type (1 - active, 0 - apparent)
        b) An y (appliance reading, ground truth) with 1 channel:
            1. Appliance power (always wit the AC type active)
        At the end, yields multiple pairs (x, y) where x and y are aligned according to their original timestamps
        and have shapes (window size, 5) and (window size, 1) respectively.
    """

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
        self.window_stride = wandb_config.window_stride
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

        # I've decided to DO NOT NORMALIZE the aggregated power
        # mains_power = (mains_power - self.mean_power) / (self.std_power + 1e-8)

        # The timestamp that index the aggregated power reading are dismembered in several additional channels
        aggregated = self.augument_with_tempoeral_features(aggregated)

        # Now that aggregated is  DataFrame, I can augment it with active vs. apparent info as well
        if ac_type_aggregated == 'active':
            aggregated['ac_type'] = 1
        else:
            aggregated['ac_type'] = 0

        # With this we drop the DatetimeIndex from the appliance power and convert it to a numpy array
        appliance_power = appliance_power.values.reshape(-1, 1)

        return aggregated, appliance_power

    def augument_with_tempoeral_features(self, aggregated_reading: pd.Series) -> pd.DataFrame:
        """
        Uses the datetime index to expand the Series into a DataFrame that will store additional temporal features
        (columns): year, month, day, hour, minute, second, day of the week. These become new columns alongside the
        original value. The index itself is kept.

        Parameters:
        - serie: pd.Series - Input Pandas Series with a DatetimeIndex

        Returns:
        - pd.DataFrame: The original Series converted to a DataFrame with additional temporal columns.
        """
        if not isinstance(aggregated_reading.index, pd.DatetimeIndex):
            raise ValueError("The input Series must have a DatetimeIndex.")

        # Extract temporal features from the DatetimeIndex
        df = aggregated_reading.to_frame(name="value")  # Convert Series to DataFrame with a column name
        # df["year"] = serie.index.year
        # df["month"] = serie.index.month
        # df["day"] = serie.index.day
        # df["day_of_week"] = serie.index.dayofweek
        df["hour"] = aggregated_reading.index.hour
        df["minute"] = aggregated_reading.index.minute
        df["second"] = aggregated_reading.index.second

        # Removes the original timestamp index
        df.reset_index(drop=True, inplace=True)

        return df

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

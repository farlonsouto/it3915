import numpy as np
import pandas as pd
from nilmtk import TimeFrame
from tensorflow.keras.utils import Sequence

from .adjustment import Augment, Balance


class TimeSeriesDataGenerator(Sequence):
    """ Generates
        a) An x (aggregated reading, input) with 1 (one) single channels:
            Mains power (mixed AC types)
        b) An y (appliance reading, ground truth) with 1 channel:
            Appliance power (always wit the AC type active)
        c)  An m mask:
            1 for positions to compute loss, 0 for others
        At the end, yields multiple tuples (x, y, m) where x, m and y are aligned according to the original timestamps
        of x (aggregated readings) and y (ground truth).
    """

    def __init__(self, dataset, buildings, appliance, normalization_params, wandb_config, is_training=True, ):
        self.dataset = dataset
        self.buildings = buildings
        self.appliance = appliance
        self.normalization_params = normalization_params
        self.window_size = wandb_config.window_size
        self.batch_size = wandb_config.batch_size
        self.max_power = wandb_config.appliance_max_power
        self.on_threshold = wandb_config.on_threshold
        self.min_on_duration = wandb_config.min_on_duration
        self.window_stride = wandb_config.window_stride
        self.wandb_config = wandb_config
        self.balance_enabled = wandb_config.balance_enabled
        self.add_artificial_activations = wandb_config.add_artificial_activations
        self.is_training = is_training
        self.masking_portion = wandb_config.masking_portion
        self.data_generator = self._data_generator()
        self.total_samples = self._count_samples()
        self.clip_value = {
            'kettle': 16,  # Captures max power of 3200W
            'fridge': 7,  # Captures max power of 400W
            'microwave': 28,  # Captures max power of 3000W
            'dish washer': 11  # Captures max power of 2500W
        }
        self.aggregated_clip_value = 13

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
                aggregated_proc, appliance_proc, mask_proc = self._process_data(mains_power, appliance_power,
                                                                                ac_type_aggregated)

                stride = self.window_size
                if self.is_training:
                    stride = self.window_stride
                for i in range(0, len(aggregated_proc) - self.window_size + 1, stride):
                    yielded_agg = aggregated_proc[i:i + self.window_size]
                    yielded_app = appliance_proc[i:i + self.window_size]
                    yielded_msk = mask_proc[i:i + self.window_size]
                    yield yielded_agg, yielded_app, yielded_msk

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
        - Tuple of processed mains, appliance power and a mask as numpy arrays
        """

        # Sort indices to avoid performance warnings
        aggregated = aggregated.sort_index()
        appliance_power = appliance_power.sort_index()

        # Remove any possible duplicated indices
        aggregated = aggregated[~aggregated.index.duplicated(keep='first')]
        appliance_power = appliance_power[~appliance_power.index.duplicated(keep='first')]

        # Align the series
        aggregated, appliance_power = aggregated.align(appliance_power, join='inner', method='pad', limit=1)

        # mask_observer.py for values to ignore
        standby_power_mask = appliance_power == 1.0
        # Apply clipping only where the mask is False
        appliance_power = appliance_power.where(standby_power_mask,
                                                appliance_power.clip(lower=self.on_threshold, upper=self.max_power))

        # Any data modification or adjustment should take place before the normalization
        if self.appliance in ['kettle', 'microwave', 'dish washer', 'washer']:
            # Handling the least used appliances: rarely ON and, when ON, for short periods.
            augment = Augment(self.wandb_config, self.normalization_params)
            balance = Balance(self.wandb_config, self.normalization_params)
            if self.add_artificial_activations:
                aggregated, appliance_power = augment.with_artificial_activations(aggregated, appliance_power)
            if self.balance_enabled:
                aggregated, appliance_power = balance.on_off_periods(aggregated, appliance_power)

        # 1. First standardize the data
        # Extracted values for better readability
        if self.wandb_config.standardize_aggregated:
            aggregated_mean = self.normalization_params['aggregated'][ac_type_aggregated]['mean']
            aggregated_std = self.normalization_params['aggregated'][ac_type_aggregated]['std']
            aggregated = (aggregated - aggregated_mean) / aggregated_std
            # 2. Then clip outliers
            aggregated = aggregated.clip(-self.aggregated_clip_value, self.aggregated_clip_value)

        if self.wandb_config.standardize_appliance:
            appliance_mean = self.normalization_params['appliance'][self.appliance]['mean']
            appliance_std = self.normalization_params['appliance'][self.appliance]['std']
            appliance_power = (appliance_power - appliance_mean) / appliance_std
            # 2. Then clip outliers
            clip_value = self.clip_value[self.appliance]
            appliance_power = appliance_power.clip(-clip_value, clip_value)

        # 3. Finally scale to [0,1] range - normalize:
        if self.wandb_config.normalize_aggregated:
            if self.wandb_config.standardize_aggregated:
                aggregated = (aggregated + self.aggregated_clip_value) / (2 * self.aggregated_clip_value)
            else:
                aggregated_min = self.normalization_params["aggregated"][ac_type_aggregated]['min']
                aggregated_max = self.normalization_params["aggregated"][ac_type_aggregated]['max']
                aggregated = (aggregated - aggregated_min) / (aggregated_max - aggregated_min)

        if self.wandb_config.normalize_appliance:
            if self.wandb_config.standardize_appliance:
                clip_value = self.clip_value[self.appliance]
                appliance_power = (appliance_power + clip_value) / (2 * clip_value)
            else:
                appl_min = self.normalization_params["appliance"][self.appliance]['min']
                appl_max = self.normalization_params["appliance"][self.appliance]['max']
                appliance_power = (appliance_power - appl_min) / (appl_max - appl_min)

        # Drops the DatetimeIndex: The indexed series turn into a numpy ndarray
        appliance_power = appliance_power.values.reshape(-1, 1)
        aggregated = aggregated.values.reshape(-1, 1)

        masked_aggregated, mask = self.apply_mask(aggregated, float(self.masking_portion),
                                                  float(self.wandb_config.mask_token))

        return masked_aggregated, appliance_power, mask

    def apply_mask(self, aggregated, masking_portion, masking_token):
        """
        Replace a given percentage of the aggregated array with a specific masking value.
        The masked positions are the ones to be considered for loss computation in MLM.

        Parameters:
        - aggregated (np.ndarray): Input array, aggregated readings
        - masking_portion (float): Percentage of elements to replace (0 <= p <= 1).
        - masking_token (scalar): Value to replace with.

        Returns:
        - np.ndarray: Modified aggregated array with replaced values.
        - np.ndarray: Mask array indicating masked positions.
        """
        if not self.wandb_config.mlm_mask or self.wandb_config.model not in ['bert', 'transformer']:
            # For either other models or no MLM, return a mask of ones, meaning to compute loss based on all data
            mask = np.ones_like(aggregated)
            assert aggregated.shape == mask.shape, "Shape mismatch between aggregated and mask"
            return aggregated, mask

        # Flatten the aggregated array for easier manipulation
        flat_aggregated = aggregated.ravel()
        num_elements = flat_aggregated.size
        num_to_replace = int(num_elements * masking_portion)

        # Handle edge cases
        if num_to_replace <= 0:
            mask = np.ones_like(aggregated)
            assert aggregated.shape == mask.shape, "Shape mismatch between aggregated and mask"
            return aggregated, mask

        # Generate random indices for replacement
        replace_indices = np.random.choice(num_elements, size=num_to_replace, replace=False)

        # Initialize a flat mask array and mark replacement indices
        flat_mask = np.zeros_like(flat_aggregated)
        flat_mask[replace_indices] = 1.0

        # Apply the masking token at the chosen indices
        flat_aggregated[replace_indices] = masking_token

        # Reshape back to the original aggregated shape
        modified_aggregated = flat_aggregated.reshape(aggregated.shape)
        mask = flat_mask.reshape(aggregated.shape)

        # Final assertion to ensure shapes match
        assert modified_aggregated.shape == mask.shape, "Shape mismatch between modified_aggregated and mask"

        return modified_aggregated, mask

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
        Arguments:
            index: position of the batch in the Sequence.
        Returns:
            A tuple of ((batch x,batch mask), batch y)
        """
        batch_x, batch_y, batch_m = [], [], []
        for _ in range(self.batch_size):
            try:
                x, y, m = next(self.data_generator)
                if self.wandb_config.model == 'seq2p':
                    midpoint = len(y) // 2
                    y = y[midpoint]

                batch_x.append(x)
                batch_y.append(y)
                batch_m.append(m)

            except StopIteration:
                # Reset the generator and fetch the next batch
                self.data_generator = self._data_generator()
                x, y, m = next(self.data_generator)

                if self.wandb_config.model == 'seq2p':
                    midpoint = len(y) // 2
                    y = y[midpoint]

                batch_x.append(x)
                batch_y.append(y)
                batch_m.append(m)

        if self.wandb_config.model in ['bert', 'transformer']:
            return np.array(batch_x), np.array(batch_y), np.array(batch_m)
        return np.array(batch_x), np.array(batch_y)

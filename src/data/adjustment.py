import numpy as np
import pandas as pd


class Augment:

    def __init__(self, wandb_config, normalization_params):
        self.on_threshold = wandb_config.on_threshold
        self.appliance_name = wandb_config.appliance
        self.min_on_duration = wandb_config.min_on_duration
        self.max_power = wandb_config.max_power
        self.normalization_params = normalization_params

    def with_artificial_activations(self, aggregated, appliance_power):
        """
        Add artificial activations to the aggregated and appliance power data.

        Parameters:
        - aggregated: pd.Series containing mains power data.
        - appliance_power: pd.Series containing appliance power data.

        Returns:
        - Tuple of updated aggregated and appliance_power Series with artificial activations.
        """

        # Calculate statistics for the real appliance data
        stats = self.normalization_params["appliance"][self.appliance_name]

        mean_power = float(stats["ON_mean"])
        std_power = float(stats["ON_std"])
        mean_on_duration = float(stats["ON_duration_mean"])
        std_on_duration = float(stats["ON_duration_std"])

        # Generate artificial activations
        artificial_activations = pd.Series(0.0, index=appliance_power.index)

        # Generate artificial ON periods
        np.random.seed(42)  # For reproducibility
        # About 25% of the time steps are populated activations
        num_activations = len(appliance_power) // (4 * int(mean_on_duration))

        for _ in range(int(num_activations)):
            # Randomly sample ON duration
            on_duration = max(1, int(np.random.normal(mean_on_duration, std_on_duration)))

            # Randomly choose a valid start index
            possible_start_indices = artificial_activations.index[:-on_duration]  # Ensure room for on_duration
            start_index = np.random.choice(possible_start_indices)

            # Explicitly create the index range for the activation
            activation_indices = artificial_activations.index[
                                 artificial_activations.index.get_loc(start_index):  # Starting position
                                 artificial_activations.index.get_loc(start_index) + on_duration
                                 # Ending position (exclusive)
                                 ]

            # Skip if the range does not match the expected duration or overlaps with existing activations
            if len(activation_indices) != on_duration or artificial_activations.loc[activation_indices].sum() > 0:
                continue

            # Generate power values for the activation
            on_values = np.random.normal(mean_power, std_power, size=len(activation_indices))
            on_values = np.clip(on_values, 0, self.max_power)  # Clip to valid power range

            # Assign the activation
            artificial_activations.loc[activation_indices] = on_values

        # Add artificial activations to both aggregated and appliance power
        aggregated += artificial_activations
        appliance_power += artificial_activations

        return aggregated, appliance_power


class Balance:

    def __init__(self, wandb_config, normalization_params):
        self.normalization_params = normalization_params
        self.on_threshold = wandb_config.on_threshold
        self.wandb_coonfig = wandb_config

    def on_off_periods(self, aggregated_power, appliance_power, target_ratio=1.0, random_state=42):
        """
        Downsample the OFF periods in a series to balance with the ON periods.

        Parameters:
        - aggregated_power: pd:series containing the aggregated power readings.
        - appliance_power: pd.Series, the input series with sparse ON periods.
        - target_ratio: float, the desired ratio of OFF to ON samples. Default is 1.0.
        - random_state: int, random seed for reproducibility.

        Returns:
        - balanced_series: pd.Series, the down-sampled and balanced series.
        """

        # Separate ON and OFF periods
        on_periods = appliance_power[appliance_power >= self.on_threshold]  # ON readings
        off_periods = appliance_power[appliance_power < self.on_threshold]  # OFF readings

        augment = Augment(self.wandb_coonfig, self.normalization_params)
        if on_periods.empty or off_periods.empty:
            # Assuming there are no ON periods, which is by far more likely
            aggregated_power, appliance_power = augment.with_artificial_activations(aggregated_power, appliance_power)
            return aggregated_power, appliance_power

        # Calculate the target number of OFF samples
        target_off_count = int(len(on_periods) * target_ratio)

        # Downsample the OFF periods
        down_sampled_off_periods = off_periods.sample(
            n=min(target_off_count, len(off_periods)),  # Ensure we don't sample more than available
            random_state=random_state
        )

        # Combine ON and down-sampled OFF periods
        balanced_appliance_power = pd.concat([on_periods, down_sampled_off_periods])

        # Restore the original order
        balanced_appliance_power = balanced_appliance_power.sort_index()

        balanced_aggregated_power = aggregated_power.loc[balanced_appliance_power.index]

        # print("----------------------- Aggregated power size: ", len(balanced_aggregated_power))

        return balanced_aggregated_power, balanced_appliance_power

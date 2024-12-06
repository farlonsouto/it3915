import json

import pandas as pd
from nilmtk import DataSet


def compute_stats(appliance_list):
    """
    Computes statistical data (mean, median, std, etc.) for active power, apparent power,
    and specified appliances across buildings.
    """

    dataset = DataSet('../datasets/ukdale.h5')

    active_power_building = {}
    apparent_power_building = {}
    appliance_power_building = {}

    for building in dataset.buildings:
        train_elec = dataset.buildings[building].elec

        # Handle mains power statistics
        train_mains = train_elec.mains()
        mains_data_frame = train_mains.load()

        all_active_power = []
        all_apparent_power = []

        for train_mains_df in mains_data_frame:
            # Check if active power is available
            if ('power', 'active') in train_mains_df.columns:
                active_power = train_mains_df[('power', 'active')]
                all_active_power.append(active_power)

            # Check if apparent power is available
            if ('power', 'apparent') in train_mains_df.columns:
                apparent_power = train_mains_df[('power', 'apparent')]
                all_apparent_power.append(apparent_power)

        # Compute stats for active power
        if all_active_power:
            combined_active_power = pd.concat(all_active_power, axis=0)

            active_power_building[str(building)] = {
                "mean": float(combined_active_power.mean()),
                "median": float(combined_active_power.median()),
                "std": float(combined_active_power.std()),
                "Quantiles": str(combined_active_power.quantile([.25, .5, .75]).values)
            }

        # Compute stats for apparent power
        if all_apparent_power:
            combined_apparent_power = pd.concat(all_apparent_power, axis=0)
            apparent_power_building[str(building)] = {
                "mean": float(combined_apparent_power.mean()),
                "median": float(combined_apparent_power.median()),
                "std": float(combined_apparent_power.std()),
                "Quantiles": str(combined_apparent_power.quantile([.25, .5, .75]).values)
            }

        # Handle appliances
        appliance_power_building[str(building)] = {}
        for appliance_name in appliance_list:
            appliance = None
            try:
                appliance = train_elec[appliance_name]
            except KeyError:
                appliance_power_building[str(building)][appliance_name] = 'Not available'
                continue
            if appliance:
                appliance_data = appliance.load()
                all_appliance_power = []

                for appliance_df in appliance_data:
                    if ('power', 'active') in appliance_df.columns:
                        appliance_power = appliance_df[('power', 'active')]
                        all_appliance_power.append(appliance_power)

                if all_appliance_power:
                    combined_appliance_power = pd.concat(all_appliance_power, axis=0)
                    appliance_power_building[str(building)][appliance_name] = {
                        "mean": float(combined_appliance_power.mean()),
                        "median": float(combined_appliance_power.median()),
                        "std": float(combined_appliance_power.std()),
                        "Quantiles": str(combined_appliance_power.quantile([.25, .5, .75]).values)
                    }

    # Return the computed dictionaries
    return active_power_building, apparent_power_building, appliance_power_building


active_building, apparent_building, appliance_building = compute_stats(['fridge', 'kettle'])
print("Aggregated Active Power Stats per building with Quantiles [25%, 50%, 75%]")
print(json.dumps(active_building, indent=4))
print("Aggregated Apparent Power Stats per building  with Quantiles [25%, 50%, 75%]:")
print(json.dumps(apparent_building, indent=4))
print("Appliance Active Power Stats per building  with Quantiles [25%, 50%, 75%]:")
print(json.dumps(appliance_building, indent=4))

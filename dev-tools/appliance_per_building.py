import sys
import json
from nilmtk import DataSet, MeterGroup
from nilmtk import HDFDataStore
import matplotlib.pyplot as plt
import pandas as pd


# Function to list available appliances in the building
def list_appliances(hdf5_file: str, building: int):
    dataset = DataSet(hdf5_file)
    elec = dataset.buildings[building].elec

    appliances_list = []
    for meter in elec.meters:
        try:
            appliance = meter.appliances[0]
            appliance_name = appliance.metadata['original_name']
        except KeyError:
            appliance_name = appliance.metadata['type']

        appliances_list.append(appliance_name)

    print(f"Available appliances in building {building}:")
    for appliance in appliances_list:
        print(appliance)


# Function to plot aggregate and selected appliances' power data
def plot_power_data(hdf5_file: str, building: int, appliances: list):
    dataset = DataSet(hdf5_file)
    elec = dataset.buildings[building].elec

    # Get the aggregate data for the building (total energy consumption)
    aggregate = elec.power_series_all_data()

    # Fetch and resample the aggregate data to hourly means for visualization
    print("Fetching aggregate data...")
    agg_data = aggregate.power_series_all_data().resample('H').mean()

    # Set up the plot
    plt.figure(figsize=(15, 8))
    plt.plot(agg_data.index, agg_data.values, label='Aggregate Power', color='black', linewidth=1)

    # For each selected appliance, fetch and plot the power data
    for appliance_name in appliances:
        meter = elec[appliance_name]

        print(f"Fetching data for appliance: {appliance_name}")
        appliance_data = meter.power_series_all_data().resample('H').mean()

        plt.plot(appliance_data.index, appliance_data.values, label=appliance_name.capitalize())

    # Customize the plot
    plt.title(f'Power consumption for selected appliances in building {building}')
    plt.xlabel('Time')
    plt.ylabel('Power (Watts)')
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()


# Example usage:
data_set_file_path = '../datasets/ukdale.h5'
building = 1
appliances_to_plot = ['fridge', 'microwave', 'dishwasher']

if len(sys.argv) > 1:
    data_set_file_path = sys.argv[1]
if len(sys.argv) > 2:
    building = int(sys.argv[2])

ukdale = DataSet(data_set_file_path)
ukdale.set_window(start='2014-04-21', end='2014-04-22')
elec = ukdale.buildings[building].elec
elec.plot()
plt.xlabel("Time")

# List all available appliances in the selected building
list_appliances(data_set_file_path, building)

# Plot data for selected appliances
plot_power_data(data_set_file_path, building, appliances_to_plot)

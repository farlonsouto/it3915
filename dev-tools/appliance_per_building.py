import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from nilmtk import DataSet

# Check if TensorFlow detects the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.sysconfig.get_build_info())


# Function to list available appliances in the building
def list_appliances(data: DataSet, building_number: int):
    appliances_list = []
    elec_aux = data.buildings[building_number].elec
    for meter in elec_aux.meters:
        for appliance in meter.appliances:
            try:
                appliances_list.append(appliance.metadata['original_name'])
            except KeyError:
                appliances_list.append(appliance.metadata['type'])
    print(f"Available appliances in building {building_number}:")
    for appliance in appliances_list:
        print(appliance)
    return appliances_list


# Function to plot specific appliances if available
def plot_appliance(data, building_number, appliance_name):
    elec = data.buildings[building_number].elec
    try:
        meter = elec[appliance_name]
        print(f"Plotting data for {appliance_name} in Building {building_number}")
        meter.plot()
        plt.title(f"{appliance_name} - Building {building_number}")
        plt.xlabel("Time")
        plt.ylabel("Power (W)")
        plt.show()
    except KeyError:
        print(f"{appliance_name} not available in Building {building_number}")


# Example usage:
data_set_file_path = '../datasets/ukdale.h5'
appliances_to_plot = ['kettle', 'fridge']

if len(sys.argv) > 1:
    data_set_file_path = sys.argv[1]

my_dataset = DataSet(data_set_file_path)
my_dataset.set_window(start="17-02-2013", end="10-10-2013")


# Loop through buildings
for building_number in [1]:  # my_dataset.buildings.keys():
    print(f"\nProcessing Building {building_number}")

    # List all available appliances in the selected building
    available_appliances = list_appliances(my_dataset, building_number)

    # Plot specified appliances if available
    for appliance in appliances_to_plot:
        if appliance in available_appliances:
            plot_appliance(my_dataset, building_number, appliance)

print("Finished processing the dataset")

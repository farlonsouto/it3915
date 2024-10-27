import sys
import warnings
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


# Example usage:
data_set_file_path = '../datasets/ukdale.h5'

building = 5
appliances_to_plot = ['fridge', 'microwave', 'dishwasher']

if len(sys.argv) > 1:
    data_set_file_path = sys.argv[1]
if len(sys.argv) > 2:
    building = int(sys.argv[2])

my_dataset = DataSet(data_set_file_path)
my_dataset.set_window(start='2014-04-21', end='2014-04-22')

# List all available appliances in the selected building
list_appliances(my_dataset, building)

elec = my_dataset.buildings[building].elec

# Suppress warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Multiple appliances")

    # Try plotting the entire dataset
    elec.plot()
    plt.xlabel("Time")
    plt.show()

print("Finished processing the dataset")

import sys

from nilmtk import DataSet
from pprint import pprint

# Load the dataset
dataset_path = 'ukdale.h5'
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
else:
    print("Defaulting to the DataSet file ukdale.h5")

dataset = DataSet(dataset_path)

# Initialize a dictionary to store appliances and the buildings they occur in
appliance_buildings = {}

# Iterate through each building in the dataset
for building_i, building in enumerate(dataset.buildings, start=1):
    elec = dataset.buildings[building].elec

    # Collect appliances and the buildings they occur in
    for meter in elec.submeters().meters:
        for appliance in meter.appliances:
            if dataset_path == 'AMPds2.h5':
                appliance_name = appliance.metadata.get('description', 'Unknown')
            else:
                appliance_name = appliance.metadata.get('original_name', 'Unknown')  # Extract appliance name or type

            if appliance_name not in appliance_buildings:
                appliance_buildings[appliance_name] = []
            appliance_buildings[appliance_name].append(building_i)

# Saving the appliances associated with more than one building numbers
useful_appliances = {}
for appliance_name in appliance_buildings:
    list_of_buildings = list(set(appliance_buildings[appliance_name]))
    if len(dataset.buildings) == 1 or len(list_of_buildings) > 2:
        useful_appliances[appliance_name] = list_of_buildings

# Pretty print the appliances and the buildings they occur in
print("Appliance-wise Buildings:")
pprint(useful_appliances)

# Close the dataset
dataset.store.close()

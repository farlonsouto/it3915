from nilmtk import DataSet

# Load the dataset
dataset_path = 'ukdale.h5'
dataset = DataSet(dataset_path)

# Initialize a dictionary to store building-wise appliances
building_appliances = {}

# Iterate through each building in the dataset
for building_i, building in enumerate(dataset.buildings, start=1):
    elec = dataset.buildings[building].elec
    appliances = set()

    # Collect unique appliance names/types for the current building
    for meter in elec.submeters().meters:
        for appliance in meter.appliances:
            appliance_name = appliance.metadata.get('original_name', 'Unknown')  # Extract appliance name or type
            appliances.add(appliance_name)

    # Store the appliances for the current building
    building_appliances[building_i] = list(appliances)

# Pretty print the building-wise appliances
print("Building-wise Appliances:")
print(building_appliances)

# Close the dataset
dataset.store.close()

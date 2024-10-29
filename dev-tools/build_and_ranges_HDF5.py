import sys
import gc
from nilmtk import DataSet
import pandas as pd

# Define the file path to your dataset
data_set_file_path = '../datasets/ukdale.h5'
if len(sys.argv) > 1:
    data_set_file_path = sys.argv[1]

# Load the dataset
my_dataset = DataSet(data_set_file_path)


def print_building_info(data: DataSet, building_number: int):
    print(f"\nBuilding {building_number} Information:")
    elec = data.buildings[building_number].elec

    # List all available appliances in the building
    appliances_list = []
    for meter in elec.meters:
        for appliance in meter.appliances:
            appliances_list.append(appliance.metadata.get('original_name', appliance.metadata.get('type')))

    print("Available appliances:")
    for appliance in appliances_list:
        print(f"  - {appliance}")

    # Load mains data with chunking
    try:
        for mains_df in elec.mains().load(chunk_size=10000):  # Adjust chunk_size as needed
            mains_start, mains_end = mains_df.index[0], mains_df.index[-1]
            print(f"  Mains data range: {mains_start} to {mains_end}")
            print(f"  Mains data columns: {list(mains_df.columns)}")
            break  # Only load first chunk to get metadata, remove this break if you need all data
    except (KeyError, IndexError, StopIteration):
        print("  No mains data available.")

    # Print date range for each appliance with chunking
    for appliance_name in appliances_list:
        try:
            appliance_meter = elec[appliance_name]
            for appliance_df in appliance_meter.load(chunk_size=4096):
                start, end = appliance_df.index[0], appliance_df.index[-1]
                print(f"  {appliance_name} data range: {start} to {end}")
                print(f"  {appliance_name} data columns: {list(appliance_df.columns)}")
                break
        except (KeyError, IndexError, StopIteration):
            print(f"  No data available for {appliance_name}.")

    # Clean up to free memory
    del appliances_list
    gc.collect()


# Loop through buildings and print information
for building_number in [1]:  # or list(my_dataset.buildings.keys())
    print_building_info(my_dataset, building_number)

print("\nFinished processing the dataset")

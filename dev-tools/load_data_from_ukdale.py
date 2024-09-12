import h5py
import numpy as np
import hdf5plugin

# Path to the dataset
dataset_path = '../datasets/ukdale.h5'


# Function to load data for a specific meter
def load_meter_data(filepath, building, meter):
    with h5py.File(filepath, 'r') as f:
        try:
            # Access the 'table' dataset
            dataset = f[f'{building}/elec/{meter}/table']
            data = dataset[:]['values_block_0']  # Extract 'values_block_0' field
            return data
        except KeyError as e:
            print(f"Error loading data for {meter}: {e}")
            return None


# Example usage
def main():
    buildings = ['building1', 'building2', 'building3', 'building4', 'building5']
    meters = ['meter1', 'meter10', 'meter11']

    for building in buildings:
        print(f"Building: {building}")
        for meter in meters:
            print(f"Processing {meter}...")
            data = load_meter_data(dataset_path, building, meter)
            if data is not None:
                print(f"Loaded data for {meter} with shape {data.shape}")
            else:
                print(f"Failed to load data for {meter}")


if __name__ == "__main__":
    main()

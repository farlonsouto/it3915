import pandas as pd
import h5py
import hdf5plugin

# Load the dataset
dataset_path = 'AMPds2.h5'

# Define the buildings and date ranges for training
train_buildings = {
    1: {
        'start_time': '2015-01-28',
        'end_time': '2019-02-12'
    }
    # ,2: {
    #   'start_time': '2015-01-28',
    #   'end_time': '2015-02-12'
    # }
    # ,3: {
    #    'start_time': '2015-04-30',
    #    'end_time': '2015-05-14'
    # }
}


# Function to check data availability
def check_data_availability(building, start_time, end_time):
    with h5py.File(dataset_path, 'r') as f:
        path = f'building{building}/elec/meter1/table'
        if path in f:
            data = f[path]
            print(f"Reading data for building {building} from {path}")
            print("Available columns:", data.dtype.names)

            # Convert the time range to pandas timestamps
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            # Read the data
            try:
                timestamps = pd.to_datetime(data['index'], unit='ns')
                # Filter the data for the specified time range
                filtered_data = data[(timestamps >= start_time) & (timestamps <= end_time)]

                if len(filtered_data) > 0:
                    print(f"Data availability from {start_time} to {end_time} for building {building}:")
                    print(filtered_data)
                else:
                    print(f"No data available from {start_time} to {end_time} for building {building}")
                    # Optionally print the available data range
                    print(f"Available data ranges for building {building}:")
                    print(
                        f"Start: {pd.to_datetime(timestamps.min(), unit='ns')}, End: {pd.to_datetime(timestamps.max(), unit='ns')}")
            except Exception as e:
                print(f"Error reading data for building {building}: {e}")
        else:
            print(f"No data for building {building}")


# Check data availability for each building in training set
for building, times in train_buildings.items():
    check_data_availability(building, times['start_time'], times['end_time'])

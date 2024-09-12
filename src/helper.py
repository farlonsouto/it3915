import datetime

import pandas as pd
import hdf5plugin
import h5py


def load_data(filepath, building, start_time, end_time):
    with h5py.File(filepath, 'r') as f:
        # Access the dataset for the specified building and meter
        mains_data = f[f'building{building}/elec/meter1/table']

        # Read index and values
        index = mains_data['index'][:]
        values = mains_data['values_block_0'][:]

        # Try to infer the correct time unit by inspecting the range of values in 'index'
        try:
            timestamps = pd.to_datetime(index, unit='s')  # Assuming Unix timestamp in seconds
        except (ValueError, OverflowError):
            # Fallback to other units if necessary
            try:
                timestamps = pd.to_datetime(index, unit='ms')  # Try milliseconds
            except (ValueError, OverflowError):
                timestamps = pd.to_datetime(index, unit='ns')  # Try nanoseconds (last resort)

        # Flatten the values array
        power = values.flatten()

        # Create DataFrame
        df = pd.DataFrame({'timestamp': timestamps, 'power': power})

        # Filter DataFrame by date range
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        # Handle cases where the DataFrame might be empty
        if df.empty:
            raise ValueError(f"No data available for the specified range: {start_time} to {end_time}")

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        return df


# Load and pre-process the data

def load_data_original(filepath, building, start_time, end_time):
    with h5py.File(filepath, 'r') as f:
        mains_data = f[f'building{building}/elec/meter1/table']
        index = mains_data['index'][:]
        values = mains_data['values_block_0'][:]
        timestamps = pd.to_datetime(index)
        power = values.flatten()
        df = pd.DataFrame({'timestamp': timestamps, 'power': power})
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        df.set_index('timestamp', inplace=True)
        return df


# Function to get the current date and time as a string
def datetime_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


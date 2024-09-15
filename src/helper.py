import datetime

import pandas as pd
import hdf5plugin
import h5py
import matplotlib.pyplot as plt
import pandas as pd


# Function to get the current date and time as a string
def datetime_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


# Function to get the time range of the dataset (start and end timestamps)
def get_time_range(filepath, building):
    with h5py.File(filepath, 'r') as f:
        # Access the main meter dataset (aggregated meter)
        mains_data = f[f'building{building}/elec/meter1/table']

        # Get the index (which likely contains timestamps)
        index = mains_data['index'][:]

        # Try to infer the correct time unit by inspecting the range of values in 'index'
        try:
            timestamps = pd.to_datetime(index, unit='s')  # Unix timestamp in seconds
        except (ValueError, OverflowError):
            try:
                timestamps = pd.to_datetime(index, unit='ms')  # Try milliseconds
            except (ValueError, OverflowError):
                timestamps = pd.to_datetime(index, unit='ns')  # Try nanoseconds

        # Extract the start and end times
        start_time = timestamps.min()
        end_time = timestamps.max()

        return start_time, end_time


# Function to list available appliances (meters) and their names in the building
def list_appliances(filepath, building):
    appliances = []
    with h5py.File(filepath, 'r') as f:
        # Access the building group
        building_group = f[f'building{building}/elec']

        # Find all available meters in the building (skip meter1 which is aggregated)
        for key in building_group.keys():
            if key.startswith('meter') and key != 'meter1':
                meter_group = building_group[key]
                # Check for metadata or some attribute that indicates the appliance name
                try:
                    # Assuming appliance names are stored in an attribute or dataset
                    appliance_name = meter_group.attrs.get('appliance_name', key)  # Fallback to meter ID if not found
                except KeyError:
                    # If no metadata or attribute, fallback to meter ID
                    appliance_name = key
                appliances.append((key, appliance_name))

    return appliances


# Function to load data for a given meter and time range
def load_data(filepath, building, meter, start_time, end_time):
    with h5py.File(filepath, 'r') as f:
        # Access the specified meter's dataset
        meter_data = f[f'building{building}/elec/{meter}/table']

        # Read index and values
        index = meter_data['index'][:]
        values = meter_data['values_block_0'][:]

        # Convert timestamps
        try:
            timestamps = pd.to_datetime(index, unit='s')
        except (ValueError, OverflowError):
            try:
                timestamps = pd.to_datetime(index, unit='ms')
            except (ValueError, OverflowError):
                timestamps = pd.to_datetime(index, unit='ns')

        # Ensure that the timestamps and values have the same length
        min_length = min(len(timestamps), len(values))
        timestamps = timestamps[:min_length]
        values = values[:min_length]

        # Create a DataFrame
        df = pd.DataFrame({'timestamp': timestamps, 'power': values.flatten()})

        # Filter the DataFrame by the date range
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        return df


# Function to plot the time series data for the aggregated meter and appliances
def plot_time_series(filepath, building, start_time, end_time):
    # Load the aggregated data (meter1)
    aggregated_df = load_data(filepath, building, 'meter1', start_time, end_time)

    with h5py.File(filepath, 'r') as f:
        building_group = f[f'building{building}/elec']

        # Create a figure for the plots
        plt.figure(figsize=(12, 8))

        # Plot aggregated meter data
        plt.subplot(2, 1, 1)
        plt.plot(aggregated_df.index, aggregated_df['power'], label='Aggregated')
        plt.title('Aggregated Meter Readings')
        plt.ylabel('Power (W)')
        plt.grid(True)

        # Plot appliances data (other meters)
        plt.subplot(2, 1, 2)
        appliances = list_appliances(filepath, building)
        for key, appliance_name in appliances:
            appliance_df = load_data(filepath, building, key, start_time, end_time)
            plt.plot(appliance_df.index, appliance_df['power'], label=appliance_name)

        plt.title('Appliance Meter Readings')
        plt.xlabel('Time')
        plt.ylabel('Power (W)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Example usage:
filepath = '../datasets/ukdale.h5'
building = 1  # Specify building number

# Get time range of the dataset
start_date, end_date = get_time_range(filepath, building)
print(f"Start Date: {start_date}, End Date: {end_date}")

# List available appliances (meters)
appliances = list_appliances(filepath, building)
print("Available Appliances:", [appliance_name for _, appliance_name in appliances])

# Plot time series for a specific time range
start_time = pd.to_datetime('2013-04-01')
end_time = pd.to_datetime('2013-04-02')
plot_time_series(filepath, building, start_time, end_time)


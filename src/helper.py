import datetime

import pandas as pd
import hdf5plugin
import h5py


# Load and pre-process the data
def load_data(filepath, building, start_time, end_time):
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


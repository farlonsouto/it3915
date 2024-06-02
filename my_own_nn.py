# Import necessary libraries
import pandas as pd
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# Function to load and preprocess data from the HDF5 file
def load_data(filepath, building, start_time, end_time):
    with h5py.File(filepath, 'r') as f:
        # Load mains (aggregate) power data
        mains_data = f[f'building{building}/elec/meter1']  # adjust meter1 as needed
        timestamps = pd.to_datetime(np.array(mains_data['time']))
        power = np.array(mains_data['power']['active'])

        # Create a DataFrame
        df = pd.DataFrame({'timestamp': timestamps, 'power': power})

        # Filter based on start and end time
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        df.set_index('timestamp', inplace=True)

        return df


# Load training and testing data
train_mains = load_data('ukdale.h5', 1, '2014-01-01', '2015-02-15')
test_mains = load_data('ukdale.h5', 5, '2014-01-01', '2015-02-15')

# Normalize the power data using the max value from the training data
max_power = train_mains['power'].max()
train_mains['power'] = train_mains['power'] / max_power
test_mains['power'] = test_mains['power'] / max_power


# Function to create the neural network model
def create_model(input_shape_param):
    model = Sequential()
    # Add Conv1D layers
    model.add(Conv1D(16, kernel_size=4, activation='relu', input_shape=input_shape_param))
    model.add(Conv1D(16, kernel_size=4, activation='relu'))
    model.add(Flatten())  # Flatten the output for the Dense layer
    model.add(Dense(128, activation='relu'))  # Fully connected layer
    model.add(Dense(1, activation='linear'))  # Output layer
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Define parameters for the TimeseriesGenerator
window_size = 60
batch_size = 32

# Create data generators for training and testing
train_generator = TimeseriesGenerator(train_mains['power'].values, train_mains['power'].values,
                                      length=window_size, batch_size=batch_size)
test_generator = TimeseriesGenerator(test_mains['power'].values, test_mains['power'].values,
                                     length=window_size, batch_size=batch_size)

# Define the input shape for the model
input_shape = (window_size, 1)
# Create the model
model = create_model(input_shape)

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model on the test data
loss, mae = model.evaluate(test_generator)
print(f'Test MAE: {mae}')

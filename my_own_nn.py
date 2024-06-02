import pandas as pd
import hdf5plugin
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Function to load and preprocess data from the HDF5 file
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


# Load training and testing data
train_mains = load_data('ukdale.h5', 1, '2014-01-01', '2015-02-15')
test_mains = load_data('ukdale.h5', 5, '2014-01-01', '2015-02-15')
max_power = train_mains['power'].max()
train_mains['power'] = train_mains['power'] / max_power
test_mains['power'] = test_mains['power'] / max_power


# Function to create the neural network model
def create_model(input_shape_param):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=4, activation='relu', input_shape=input_shape_param))
    model.add(Conv1D(16, kernel_size=4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Define parameters for the TimeseriesGenerator
window_size = 60
batch_size = 32

# Reshape the data to have a third dimension
train_mains_reshaped = train_mains['power'].values.reshape(-1, 1)
test_mains_reshaped = test_mains['power'].values.reshape(-1, 1)

# Create data generators for training and testing
train_generator = TimeseriesGenerator(train_mains_reshaped, train_mains_reshaped,
                                      length=window_size, batch_size=batch_size)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped,
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

# Generate predictions on test data
predictions = model.predict(test_generator)

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))
original_line, = ax.plot([], [], label='Original Aggregate Power', color='blue')
predicted_lines = [ax.plot([], [], label=f'Predicted Appliance {i + 1}')[0] for i in range(predictions.shape[1])]


# Define the update function for animation
def update(frame):
    # Plot original aggregated power consumption up to the current timestamp
    original_line.set_data(test_mains.index[:frame], test_mains['power'][:frame])

    # Plot predicted power consumption for each appliance up to the current timestamp
    for i, line in enumerate(predicted_lines):
        line.set_data(test_mains.index[:frame], predictions[:frame, i])

    # Update the plot title
    ax.set_title(f'Disaggregated Power Consumption (Up to Timestamp: {test_mains.index[frame]})')

    return original_line, *predicted_lines


# Create animation
ani = FuncAnimation(fig, update, frames=(len(test_mains) - window_size), interval=100, blit=True)

plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.show()

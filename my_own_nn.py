import pandas as pd
import hdf5plugin
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt


# Function to load and preprocess data from the HDF5 file
def load_data(filepath, building, start_time, end_time):
    with h5py.File(filepath, 'r') as f:
        # Adjust the dataset path based on your HDF5 file structure
        mains_data = f[f'building{building}/elec/meter1/table']

        # Extract data
        index = mains_data['index'][:]  # Load entire dataset into memory
        values = mains_data['values_block_0'][:]  # Load entire dataset into memory

        # Print out some sample values for inspection
        print(f"Index sample: {index[:10]}")
        print(f"Min index value: {index.min()}")
        print(f"Max index value: {index.max()}")

        # Convert to appropriate format
        timestamps = pd.to_datetime(index)  # Default unit is ns, which is appropriate here
        power = values.flatten()  # Ensure 'values_block_0' is the correct field

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
loss, mae = model.evaluate(test_generator, verbose=0)
print(f'Test MAE: {mae}')

# Get the predictions for each window in the test data
predictions = model.predict(test_generator)

# Plot original aggregated power consumption
plt.figure(figsize=(10, 6))
plt.plot(test_mains.index, test_mains['power'], label='Original Aggregate Power', color='blue')

# Plot predicted power consumption for each appliance
for i in range(predictions.shape[1]):
    plt.plot(test_mains.index[window_size:], predictions[:, i], label=f'Predicted Appliance {i+1}', alpha=0.7)

plt.title('Disaggregated Power Consumption')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.show()
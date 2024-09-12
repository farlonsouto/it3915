import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import hdf5plugin
from tensorflow.keras.models import load_model
from matplotlib.animation import FuncAnimation


# Helper functions to load data
def load_meter_data(filepath, building, meter):
    with h5py.File(filepath, 'r') as f:
        # Access the dataset 'table' for the given meter
        try:
            dataset = f[f'{building}/elec/{meter}/table']
            data = dataset[:]['values_block_0']  # Extract 'values_block_0' field
        except KeyError as e:
            print(f"Error loading data for {meter}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading data for {meter}: {e}")
            return None

    return data


# Set up the file path and appliance meters
dataset_path = '../datasets/ukdale.h5'
building = 'building1'
appliance_meters = ['meter1', 'meter10', 'meter11']  # Adjusted meters based on available ones

# Load data for each meter
meter_data = {}
for meter in appliance_meters:
    print(f"Processing {meter}...")
    try:
        meter_data[meter] = load_meter_data(dataset_path, building, meter)
    except Exception as e:
        print(f"Error loading data for {meter}: {e}")

# Load pre-trained model
model_path = '../models/seq2point_cnn_vanilla.keras'
model = load_model(model_path)

# Create animated plots comparing ground truth with predicted values
fig, ax = plt.subplots()


# Prepare animation function
def update(frame):
    ax.clear()

    # For simplicity, we'll only plot one meter at a time
    meter = appliance_meters[frame % len(appliance_meters)]
    data = meter_data.get(meter, [])[:1000]  # Take the first 1000 samples

    if len(data) == 0:
        ax.set_title(f'No data available for {meter}')
        return

    # Reshape input for the model
    if len(data) < 60:
        ax.set_title(f'Not enough data for {meter}')
        return

    input_data = np.expand_dims(np.array(data[:60]), axis=(0, -1))  # Sequence length of 60
    predicted = model.predict(input_data)

    # Plot ground truth vs prediction
    ax.plot(data[:1000], label=f'Ground Truth - {meter}')
    ax.plot(np.arange(60, 60 + len(predicted)), predicted.flatten(), label='Predicted', color='red')

    ax.set_title(f'Meter: {meter}')
    ax.legend()


# Create animation
anim = FuncAnimation(fig, update, frames=len(appliance_meters), repeat=True, interval=2000)

# Show the plot
plt.show()

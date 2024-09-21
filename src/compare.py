import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.models import load_model
import numpy as np


# Function to load test data for the selected building and appliance using NILMTK
def load_test_data(building_number: int, appliance, start_time, end_time):
    # Path to the .h5 dataset file
    dataset_file_path = '../datasets/ukdale.h5'  # Replace with the actual path to your .h5 file

    # Load the dataset using NILMTK
    dataset = DataSet(dataset_file_path)

    # Set the window for the data (this limits the data to the time range you are interested in)
    dataset.set_window(start=start_time, end=end_time)

    # Get the electricity data for the selected building
    elec = dataset.buildings[building_number].elec

    # Find the specific appliance you want to compare
    appliance_meter = elec[appliance]

    # Load the mains data and appliance data (ground truth)
    mains_power = elec.mains().load().next()  # Aggregate mains data
    appliance_power = appliance_meter.load().next()  # Appliance-specific data

    # Assuming the 'power_active' column contains the data you need
    test_data = mains_power['power_active'].values
    ground_truth = appliance_power['power_active'].values

    # Normalize the data by dividing by the maximum value (optional, depending on your model)
    max_power = test_data.max()
    test_data = test_data / max_power
    ground_truth = ground_truth / max_power

    # Reshape the test data to match the model input shape (150 time steps, 1 feature)
    test_data = test_data.reshape(-1, 150, 1)

    return test_data, ground_truth


# Function to get predictions from the model
def get_predictions(model, test_data):
    return model.predict(test_data)


# Function to plot the results
def plot_results(ground_truth, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth, label='Ground Truth', color='red')
    plt.plot(predictions, label='Predictions', color='green')
    plt.xlabel('Time')
    plt.ylabel('Power Consumption')
    plt.title('Ground Truth vs Model Prediction')
    plt.legend()
    plt.show()


# Function to compare data based on user inputs
def compare_data():
    building = building_var.get()
    appliance = appliance_var.get()
    start_time = pd.to_datetime(start_time_entry.get())
    end_time = pd.to_datetime(end_time_entry.get())

    # Load test data for the selected building, appliance, and time range
    test_data, ground_truth = load_test_data(int(building), appliance, start_time, end_time)

    # Get model predictions
    predictions = get_predictions(model, test_data)

    # Plot the results
    plot_results(ground_truth, predictions.flatten())


# Load the model (assuming you have already restored variables and .pb file)
model = load_model('/home/fsouto/ntnu/sandbox/nilm/src/models')

# GUI Setup
root = tk.Tk()
root.title("Appliance Power Prediction Comparison")

# Building selection
tk.Label(root, text="Select Building:").grid(row=0, column=0)
building_var = tk.StringVar()
building_dropdown = ttk.Combobox(root, textvariable=building_var)
building_dropdown['values'] = ['1', '5']  # Building 1 and Building 5
building_dropdown.grid(row=0, column=1)

# Appliance selection
tk.Label(root, text="Select Appliance:").grid(row=1, column=0)
appliance_var = tk.StringVar()
appliance_dropdown = ttk.Combobox(root, textvariable=appliance_var)
appliance_dropdown['values'] = ['fridge', 'microwave', 'dishwasher']  # Example appliances
appliance_dropdown.grid(row=1, column=1)

# Start time input
tk.Label(root, text="Start Time (YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0)
start_time_entry = tk.Entry(root)
start_time_entry.grid(row=2, column=1)

# End time input
tk.Label(root, text="End Time (YYYY-MM-DD HH:MM:SS):").grid(row=3, column=0)
end_time_entry = tk.Entry(root)
end_time_entry.grid(row=3, column=1)

# Compare button
compare_button = tk.Button(root, text="Compare", command=compare_data)
compare_button.grid(row=4, column=0, columnspan=2)

# Start the GUI loop
root.mainloop()

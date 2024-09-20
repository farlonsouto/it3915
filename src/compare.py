import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np


# Function to load test data for the selected building and appliance
def load_test_data(building, appliance, start_time, end_time):
    # Replace these lines with code that loads your actual test data
    # Assume the data is stored in a CSV or similar format for now.
    file_path = f"data/building_{building}_{appliance}.csv"  # Example file path
    data = pd.read_csv(file_path, parse_dates=['timestamp'])

    # Filter data by the selected time interval
    data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]

    # Assuming the data has 'mains_power' for input and 'appliance_power' for ground truth
    test_data = data['mains_power'].values
    ground_truth = data['appliance_power'].values

    # Reshape the test_data to match the model input shape (150 time steps, 1 feature)
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
    test_data, ground_truth = load_test_data(building, appliance, start_time, end_time)

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

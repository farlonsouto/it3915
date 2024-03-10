# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nilmtk

from nilmtk import DataSet
import matplotlib.pyplot as plt
import numpy as np

# Load REDD dataset
redd = DataSet('/path/to/redd/low_freq')

# Select a building and load mains and appliance data
building = 1
elec = redd.buildings[building].elec

# Get appliances
appliances = elec.appliances

# Select appliances to plot (change as needed)
appliances_to_plot = ['fridge', 'microwave', 'dish washer']

# Plot settings
colors = ['blue', 'red', 'green']
num_appliances = len(appliances_to_plot)

# Plot each selected appliance
plt.figure(figsize=(12, 8))
for i, app_name in enumerate(appliances_to_plot):
    app = appliances[app_name]
    app_data = app.power_series_all_data(sample_period=60).dropna()  # Resample data if needed
    time_index = app_data.index
    app_power = app_data.values
    plt.plot(time_index, app_power, color=colors[i], label=app_name)

plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.title('Appliance Power Consumption Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

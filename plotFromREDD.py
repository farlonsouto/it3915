import nilmtk
from nilmtk import DataSet
import matplotlib.pyplot as plt

# Load REDD dataset
redd = DataSet('ukdale.h5')

# Select a building and load mains and appliance data
building = 1
elec = redd.buildings[building].elec

# Get appliances into a dictionary
appliances = {}
for app in elec.appliances:
    metadata = app.metadata
    original_name = metadata.get('original_name')
    app_type = metadata.get('type')
    if original_name is not None:
        appliances[original_name] = app
    elif app_type is not None:
        appliances[app_type] = app
    else:
        print("Metadata missing both 'original_name' and 'type' for this appliance:", metadata)

print("Appliances in dataset:", list(appliances.keys()))

# Select appliances to plot (change as needed)
appliances_to_plot = ['fridge', 'microwave', 'dishwasher']

# Plot settings
colors = ['blue', 'red', 'green']
num_appliances = len(appliances_to_plot)

# Plot each selected appliance
plt.figure(figsize=(12, 8))
for i, app_name in enumerate(appliances_to_plot):
    app = appliances.get(app_name)
    if app is None:
        print(f"Appliance '{app_name}' not found.")
        continue

    print(f"Fetching power data for appliance '{app_name}'...")
    # Retrieve power data

    # Inside the for loop for plotting each appliance
    meters = app.meters()  # Get all meters associated with the appliance
    if meters:
        meter = meters[0]  # Assuming there's only one meter associated with the appliance
        power_data = meter.power_series_all_data(ac_type='active', sample_period=60)
    else:
        print(f"No meters found for appliance '{app_name}'.")

    # power_data = app.get_values(ac_type='active', sample_period=60)
    """
    power_data = None
    if hasattr(app, 'power_series_all_data'):
        power_data = app.power_series_all_data(sample_period=60)
    elif hasattr(app, 'power_series'):
        power_data = app.power_series(sample_period=60)
    """
    if power_data is not None:
        app_data = power_data.dropna()
        time_index = app_data.index
        app_power = app_data.values
        plt.plot(time_index, app_power, color=colors[i], label=app_name)
        print(f"Power data retrieved for appliance '{app_name}'.")
    else:
        print(f"Failed to retrieve power data for appliance '{app_name}'.")

plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.title('Appliance Power Consumption Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

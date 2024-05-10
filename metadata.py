from nilmtk import DataSet, Appliance, HDFDataStore
import pandas as pd
import matplotlib.pyplot as plt
from pandas import HDFStore, DataFrame

redd = DataSet('ukdale.h5')

# Select a building and load mains and appliance data
building = 1
elec = redd.buildings[building].elec

appliancesToPlot = ['fridge', 'microwave', 'dishwasher']
applianceData = {}
for meter in elec.meters:
    name = "Unknown Appliance"
    try:
        name = meter.appliances[0].metadata['original_name']
    except KeyError:
        name = meter.appliances[0].metadata['type']
    finally:
        if name in appliancesToPlot:
            print("Getting data for the following appliance:", name)
            applianceData[name] = meter.store[meter.metadata['data_location']]

# Initialize plot
plt.figure(figsize=(10, 5))

# Loop through applianceData dictionary
for i, (appliance_name, consumption_history) in enumerate(applianceData.items()):
    data_frame: DataFrame = consumption_history
    # Plot data
    plt.plot(data_frame.index, consumption_history['power']['active'], label=appliance_name, color=plt.cm.tab10(i))

# Set plot labels and title
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.title('Power Consumption History of Appliances')
plt.legend()
plt.grid(True)
plt.show()

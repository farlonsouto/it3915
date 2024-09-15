import sys
import json
from nilmtk import DataSet
from dataclasses import dataclass


@dataclass
class DataDescription:
    appliance_name: str
    first_read: str
    last_read: str
    sampling_frequency: float  # in hertz


def describe_data(hdf5_file: str, building_or_house: int, appliances: list) -> list:
    # Load dataset
    dataset = DataSet(hdf5_file)
    elec = dataset.buildings[building_or_house].elec

    description = []
    for meter in elec.meters:
        name = "Unknown Appliance"
        try:
            name = meter.appliances[0].metadata['original_name']
        except KeyError:
            name = meter.appliances[0].metadata['type']
        finally:
            if name in appliances:
                print(f"Analyzing data for the following appliance: {name}")

                # Get the data
                data_frame = meter.store[meter.metadata['data_location']]

                # Calculate the sampling frequency
                time_deltas = data_frame.index.to_series().diff().dropna()
                average_time_delta = time_deltas.mean()
                sampling_frequency_hz = 1 / average_time_delta.total_seconds()

                # Get the time interval
                start_time = data_frame.index.min().to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
                end_time = data_frame.index.max().to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')

                # Store the description
                description.append(DataDescription(name, start_time, end_time, sampling_frequency_hz))

    return description


appliances_to_plot = ['fridge', 'microwave', 'dishwasher']
building = 1
data_set_file_path = '../datasets/ukdale.h5'
if len(sys.argv) > 1:
    data_set_file_path = sys.argv[1]
if len(sys.argv) > 2:
    building = int(sys.argv[2])
if len(sys.argv) == 1:
    print("Defaulting to the DataSet file ukdale.h5 and to building number 1")

data_description = describe_data(data_set_file_path, building, appliances_to_plot)
for data in data_description:
    print(json.dumps(data.__dict__, indent=4))

from nilmtk import DataSet
from nilmtk.utils import print_dict


def inspect_dataset(file_path: str, target_appliance: str = "kettle"):
    # Load the dataset
    dataset = DataSet(file_path)

    # Print the dataset metadata
    print(dataset.metadata)

    # Print the building and meter information
    print("\nInspecting dataset:", file_path)
    for building in dataset.buildings:
        print(f"\nBuilding {building}:")
        # Iterate over the meters in each building
        for meter in dataset.buildings[building].elec.meters:
            print(f"  Meter: {meter}")
            # Extract a sample of the data
            data_frames_generator = meter.load()

            # Obtains aDataFrame
            for dataFrame in data_frames_generator:
                df = dataFrame #.fillna(method='ffill').fillna(method='bfill')
                # Display structure information
                # Will ignore meters with no appliances attached to it
                if not df.empty:

                    appliance_name = "Unknown Appliance"
                    if meter.is_site_meter():
                        appliance_name = "Site Meter Aggregated Readings"
                    elif len(meter.appliances) > 0:
                        appliance_name = meter.appliances[0].label()

                    if meter.is_site_meter() or (target_appliance in appliance_name):
                        print(f"\n  Appliance [{appliance_name}] :")
                        print(f"  Shape: {df.shape}")
                        print(f"  Columns: {df.columns.tolist()}")
                        print(f"  Data Types:\n{df.dtypes}")
                        data = df.get([('power', 'apparent')])
                        if data is None:
                            data = df.get([('power', 'active')])
                        print(f"  First 5 rows:\n{data.head()}")
                else:
                    print(f"  No data available for {meter}.")


# Specify the path to the .h5 file
file_path = "../datasets/AMPds2.h5"  # "../datasets/ukdale.h5"  # Change this to your actual file path

# Inspect the dataset
inspect_dataset(file_path)

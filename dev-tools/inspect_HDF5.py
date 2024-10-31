from nilmtk import DataSet


def inspect_dataset(file_path):
    # Load the dataset
    dataset = DataSet(file_path)

    # Print the building and meter information
    print("\nInspecting dataset:", file_path)
    for building in dataset.buildings:
        print(f"\nBuilding {building}:")
        # Iterate over the meters in each building
        for meter in dataset.buildings[building].elec.meters:
            print(f"  Meter: {meter}")
            # Extract a sample of the data
            dataFramesGenerator = meter.load()  # Adjust dates as needed

            # Create a DataFrame for better visualization
            for dataFrame in dataFramesGenerator:
                # Display structure information
                # Will ignore meters with no appliances attached to it
                if not dataFrame.empty and len(meter.appliances) > 0:
                    try:
                        appliance_name = meter.appliances[0].metadata['original_name']
                    except KeyError:
                        appliance_name = meter.appliances[0].metadata['type']
                    print(f"\n  Appliance [{appliance_name}] :")
                    print(f"  Shape: {dataFrame.shape}")
                    print(f"  Columns: {dataFrame.columns.tolist()}")
                    print(f"  Data Types:\n{dataFrame.dtypes}")
                    data = dataFrame.get([('power', 'apparent')])
                    if data is None:
                        data = dataFrame.get([('power', 'active')])
                    print(f"  First 5 rows:\n{data.head()}")
                else:
                    print(f"  No data available for {meter}.")


# Specify the path to the .h5 file
file_path = "../datasets/ukdale.h5"  # Change this to your actual file path

# Inspect the dataset
inspect_dataset(file_path)

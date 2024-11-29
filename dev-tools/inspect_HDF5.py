from nilmtk import DataSet


def inspect_dataset(file_path: str, target_appliances: list):
    # Load the dataset
    dataset = DataSet(file_path)

    # Print the timeframe for each building
    for building in dataset.buildings:
        print(f"Building {building}: {dataset.buildings[building].metadata['timeframe']}")

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
                df = dataFrame  # .fillna(method='ffill').fillna(method='bfill')
                # Display structure information
                # Will ignore meters with no appliances attached to it
                if not df.empty:

                    appliance_name = "Unknown Appliance"
                    if meter.is_site_meter():
                        appliance_name = "Site Meter Aggregated Readings"
                        print("Building {} - Available AC type: {}".format(building, meter.available_ac_types('power')))
                        print("Building {} - Available Columns: {}".format(building, meter.available_columns()))
                        print("Building {} - Meter model: {}".format(building, meter.device['model']))
                        print("Building {} - Meter measurements: {}".format(building, meter.device['measurements']))
                    elif len(meter.appliances) > 0:
                        appliance_name = meter.appliances[0].label()

                    is_target_appliance = False
                    for app in target_appliances:
                        if app in appliance_name:
                            is_target_appliance = True

                    if meter.is_site_meter() or is_target_appliance:
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
file_path = "../datasets/ukdale.h5"  # "../datasets/ukdale.h5"  # Change this to your actual file path

# Inspect the dataset
inspect_dataset(file_path, ["kettle", "fridge"])

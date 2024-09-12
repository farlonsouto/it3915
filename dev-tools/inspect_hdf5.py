import h5py

# Path to the dataset
dataset_path = '../datasets/ukdale.h5'


# Function to inspect the structure of the dataset
def inspect_dataset(filepath):
    with h5py.File(filepath, 'r') as f:
        # Recursively explore the structure of the file
        def explore_group(group, indent=0):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    print(f'{"  " * indent}Group: {key}')
                    explore_group(item, indent + 1)  # Recursively explore this group
                elif isinstance(item, h5py.Dataset):
                    print(f'{"  " * indent}Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}')
                else:
                    print(f'{"  " * indent}Unknown: {key}')

        print("Inspecting dataset structure...")
        explore_group(f)


# Call the inspection function
inspect_dataset(dataset_path)

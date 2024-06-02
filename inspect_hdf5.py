import h5py

filepath = 'ukdale.h5'
with h5py.File(filepath, 'r') as f:
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")


    f.visititems(print_attrs)


# Open the HDF5 file and inspect the contents of the 'table' dataset
def inspect_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        table = f['building1/elec/meter1/table']
        print("Table dtype:", table.dtype)
        print("Table shape:", table.shape)
        print("Table attributes:", list(table.attrs.keys()))

        # Read the first few rows in smaller chunks
        for i in range(5):
            print(f"Row {i}: {table[i]}")


# Inspect the datasets within the 'table' group
inspect_hdf5('ukdale.h5')

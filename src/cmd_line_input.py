import sys


def get_appliance_arg():
    appliance = 'fridge'
    if len(sys.argv) > 1:
        appliance = sys.argv[1]
        if appliance in ['kettle', 'fridge', 'washer', 'microwave', 'dish washer']:
            pass
        else:
            print(f"Invalid appliance name: {appliance}. Using fridge as default.")
    return appliance

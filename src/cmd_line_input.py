import sys


def get_args():
    appliance = None
    model = None
    args = [sys.argv[0:]]

    for model_name in ['bert', 'seq2seq', 'seq2p']:
        if model_name in args[0]:
            model = model_name

    if model is None:
        model = "bert"
        print(f"Invalid model name. Using bert as default.")

    for appliance_name in ['kettle', 'fridge', 'washer', 'microwave', 'dish washer']:
        if appliance_name in args[0]:
            appliance = appliance_name

    if appliance is None:
        appliance = "fridge"
        print(f"Invalid appliance name. Using fridge as default.")

    return model, appliance

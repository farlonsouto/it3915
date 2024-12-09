# Initialize WandB for tracking

# The article doesn't specify a single on_threshold value for all appliances. Instead, it mentions different
# on-thresholds for various appliances in Table 1. For example:

# - Fridge: 50W
# - Washer: 20W
# - Microwave: 200W
# - Dishwasher: 10W
# - Kettle: 2000W

# These values are specific to each appliance and are used to determine when an appliance is considered to be in the
# "on" state. In the context of the 'kettle' appliance that was used in the example, the correct on_threshold should be
# 2000W, not 50W. To correct this, we should modify the WandB configuration in the runner script to use the appropriate
# on_threshold for each appliance:


config = {

    # Appliance and Dataset specific
    "appliance": "kettle",  # The selected appliance must be the same for training and testing !!
    "on_threshold": 2000,
    "max_power": 3200,

    # Training
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "loss": "bert4nilm_loss",  # The BERT4NILM custom loss is called from inside the model
    "lambda_val": 1.0,  # inside the loss function
    "num_features": 1,  # The aggregated power readings, AC type; hour, minute, second; appliance status, etc

    # Input
    "window_size": 100,  # for UK Dale, 10 time steps mean 1 minute
    "masking_portion": 0.2,
    "window_stride": 10,

    # 1D Convolution layer
    "conv_kernel_size": 5,
    "conv_strides": 1,  # to be fixed in 1
    "conv_padding": 2,
    "conv_activation": "relu",  # preferably ReLU

    # Transformer
    "hidden_size": 128,
    "num_heads": 2,
    "n_layers": 1,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-6,  # Original value is 1e-6
    "dense_activation": "gelu",  # Originally GELU

    # Deconvolution layer
    "deconv_kernel_size": 4,
    "deconv_strides": 2,
    "deconv_padding": 1,
    "deconv_activation": "relu",

    # Feed-forward network dimension
    "ff_dim": 128,

    # Dimension (number of features) in the output layer
    "output_size": 1,
}


def for_appliance(appliance) -> dict:
    config["appliance"] = appliance
    if appliance == "kettle":
        config["on_threshold"] = 2000
        config["max_power"] = 3200
        config["lambda_val"] = 1.0
    elif appliance == "fridge":
        config["on_threshold"] = 50
        config["max_power"] = 400
        config["lambda_val"] = 1e-6
    elif appliance == "washer":
        config["on_threshold"] = 20
        config["max_power"] = 2500
        config["lambda_val"] = 1e-2
    elif appliance == "microwave":
        config["on_threshold"] = 200
        config["max_power"] = 3000
        config["lambda_val"] = 1.0
    elif appliance == "dishwasher":
        config["on_threshold"] = 10
        config["max_power"] = 2500
        config["lambda_val"] = 1.0

    return config

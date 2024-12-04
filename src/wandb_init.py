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
    "appliance": "fridge",  # The selected appliance must be the same for training and testing !!
    "on_threshold": 50,
    "max_power": 300,
    "min_on_duration": 60,  # in seconds
    "min_off_duration": 12,  # in seconds

    # Training
    "batch_size": 32,
    "epochs": 2,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "loss": "bert4nilm_loss",  # The BERT4NILM custom loss is called from inside the model
    "tau": 1.0,
    "lambda_val": 1,  # inside the loss function

    # Input
    "window_size": 240,  # for UK Dale, 10 time steps mean 1 minute
    "masking_portion": 0.25,

    # 1D Convolution layer
    "conv_kernel_size": 5,
    "conv_strides": 1,  # to be fixed in 1
    "conv_padding": 2,
    "conv_activation": "relu",  # preferably ReLU

    # Transformer
    "hidden_size": 256,
    "num_heads": 2,
    "n_layers": 2,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-6,  # Original value is 1e-6
    "dense_activation": "gelu",  # Originally GELU

    # Deconvolution layer
    "deconv_kernel_size": 4,
    "deconv_strides": 2,
    "deconv_padding": 1,
    "deconv_activation": "relu",

    # Feed-forward network dimension
    "ff_dim": 197,

    # Dimension (number of features) in the output layer
    "output_size": 1,
}

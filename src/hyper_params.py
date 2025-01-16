# Initialize WandB for tracking

config = {

    # Training
    "batch_size": 32,  # Larger for better gradient estimates
    "epochs": 5,
    "learning_rate": 1e-4,  # Higher learning rate with warmup
    "optimizer": "adam",
    "loss": "bert4nilm_loss",  # "mse" or "huber" seems to make no difference
    "temperature": 0.1,
    "lambda_val": 1.0,  # inside the bert4nilm loss function
    "num_features": 1,  # The aggregated power readings, AC type; hour, minute, second; appliance status, etc
    "continuation": False,  # If it is the continuation of a previous training. True at the MLM 2nd run, without mask

    # Input
    "window_size": 512,  # for UK Dale, 10 time steps mean 1 minute
    "window_stride": 1,
    "mlm_mask": True,  # MLM masking for BERT
    "mask_token": -3100.0,
    "masking_portion": 0.25,
    "add_artificial_activations": False,
    "balance_enabled": False,
    "standardize_aggregated": True,  # z-score, Uses mean and std: x = (x - x_mean) / x_std
    "standardize_appliance": False,  # z-score, Uses mean and std: y = (y - y_mean) / y_std

    # 1D Convolution layer
    "conv_kernel_size": 5,
    "conv_strides": 1,  # to be fixed in 1
    "conv_padding": 2,
    "conv_activation": "relu",  # preferably ReLU

    # Transformer
    "hidden_size": 256,  # Reduced to allow for more layers within same compute
    "num_heads": 2,  # More heads to capture different pattern aspects
    "num_layers": 2,  # Also for seq2seq LSTMs
    "ff_dim": 128,  # 4x hidden_size is the recommended
    "dropout": 0.1,  # Applies also to the seq2seq LSTM
    "layer_norm_epsilon": 1e-6,  # Original value is 1e-6
    "dense_activation": "gelu",

    # Kernel regularization
    "kernel_regularizer": "None",  # For all architectures

    # Deconvolution layer
    "deconv_kernel_size": 4,
    "deconv_strides": 2,
    "deconv_padding": 1,
    "deconv_activation": "linear",

    # Dimension (number of features) in the output layer
    "output_size": 1,
}


def for_model_appliance(model_name, appliance_name) -> dict:
    # Initialize the configuration dictionary
    config["appliance_name"] = appliance_name
    config["model"] = model_name

    # Set the appliance-specific configuration
    if appliance_name == "kettle":
        config.update({
            "lambda_val": 1.0,
            "appliance_max_power": 3100.00,
            "on_threshold": 2000.0,
            "min_on_duration": 12,
            "min_off_duration": 0,
        })
    elif appliance_name == "fridge":
        config.update({
            "lambda_val": 1e-6,
            "appliance_max_power": 400,
            "on_threshold": 50,
            "min_on_duration": 60,
            "min_off_duration": 12,
        })
    elif appliance_name == "washer":
        config.update({
            "lambda_val": 1e-2,
            "appliance_max_power": 2500,
            "on_threshold": 20,
            "min_on_duration": 1800,
            "min_off_duration": 160,
        })
    elif appliance_name == "microwave":
        config.update({
            "lambda_val": 1.0,
            "appliance_max_power": 3000,
            "on_threshold": 200,
            "min_on_duration": 12,
            "min_off_duration": 30,
        })
    elif appliance_name == "dish washer":
        config.update({
            "lambda_val": 1.0,
            "appliance_max_power": 2500,
            "on_threshold": 10,
            "min_on_duration": 1800,
            "min_off_duration": 1800,
        })
    else:
        raise ValueError(f"Unknown appliance: {appliance_name}")

    return config

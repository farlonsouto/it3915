# Import necessary libraries
import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import wandb
from wandb.integration. keras import WandbMetricsLogger, WandbModelCheckpoint
import random

'''
1. **Hyperparameter Choices**:
   - **`head_size=32`**: The size of each attention head. A larger head size captures more detailed relationships but comes with a higher computational cost.
   - **`num_heads=2`**: The number of attention heads. Multiple heads allow the model to capture different types of relationships in parallel. Two heads will reduce complexity.
   - **`ff_dim=64`**: The dimension of the feed-forward network. A smaller dimension reduces complexity.
   - **`num_transformer_blocks=2`**: The number of stacked transformer blocks. Fewer blocks reduce the computational load.
   - **`dropout=0.2`**: A higher dropout value helps prevent overfitting by randomly disabling some neurons during training.

2. **Alternatives**:
   - **Head Size**: You could increase the head size (e.g., to 64) to capture more intricate details in the data, though this might require more computational resources.
   - **Number of Transformer Blocks**: Increasing to 4 or more blocks could help with deeper learning, though overfitting may become a concern.
   - **Dropout Rate**: You could experiment with a higher dropout rate if overfitting becomes a problem during training.

3. **Transformer Architecture**:
   - **Multi-Head Attention**: This layer allows the model to focus on different parts of the input sequence (power consumption at different time steps) simultaneously, making it very effective for capturing long-term dependencies.
   - **Feed Forward Network**: After the attention layer, a feed-forward network is applied to each position independently, adding non-linearity and increasing model capacity.
   - **Layer Normalization**: This helps stabilize training by normalizing the inputs across the batch, preventing the model from becoming too sensitive to certain input values.
   - **Residual Connections**: By adding the input to the output of the attention and feed-forward layers, the model can retain information from earlier layers, preventing the loss of information during deep training.

4. **Callbacks**:
   - **Learning Rate Scheduler**: This callback reduces the learning rate during training to help the model converge smoothly.
   - **Early Stopping**: It helps prevent overfitting by stopping training if the validation loss stops improving.
   - **Model Checkpoint**: Ensures the best model (with the lowest validation loss) is saved, even if training continues to overfit later.
'''

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="nilm_transformer",

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "dropout": random.uniform(0.01, 0.80),
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 8,
        "batch_size": 256
    }
)

# [optional] use wandb.config as your config
config = wandb.config

# Load data using NILMTK
# We are training the model on data from Building 1, from '2014-01-01' to '2015-02-15'
# The data contains power readings for multiple appliances over time.
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-01', end='2015-02-15')

# NILM involves breaking down aggregate power signals (like mains) into individual appliance signals.
# 'train_elec' gets the electrical data for training, and 'train_mains' extracts mains readings from that data.
train_elec = dataset.buildings[1].elec
train_mains = train_elec.mains()

# Similarly, we load test data from another building (Building 5).
test_elec = dataset.buildings[5].elec
test_mains = test_elec.mains()

# Load the mains data into pandas DataFrames.
# This gives us a time series of power readings that we'll feed into the model.
train_mains_df = next(train_mains.load())
test_mains_df = next(test_mains.load())

# If the columns in the DataFrame have a MultiIndex (which can happen with NILMTK data), we flatten it.
# This ensures the columns are simpler to handle.
if isinstance(train_mains_df.columns, pd.MultiIndex):
    train_mains_df.columns = ['_'.join(col).strip() for col in train_mains_df.columns.values]
if isinstance(test_mains_df.columns, pd.MultiIndex):
    test_mains_df.columns = ['_'.join(col).strip() for col in test_mains_df.columns.values]

# Access the 'power_active' column, which contains the actual power consumption data.
train_mains_power = train_mains_df['power_active']
test_mains_power = test_mains_df['power_active']

# Normalize the data by dividing by the maximum value.
# This is a common preprocessing step in neural networks to ensure that the data is scaled,
# preventing large values from overwhelming the model.
max_power = train_mains_power.max()
train_mains_power = train_mains_power / max_power
test_mains_power = test_mains_power / max_power

# Reshape the data into the form expected by the model.
# Here, we're transforming the time series data into a 2D array where each row is a time step.
train_mains_reshaped = train_mains_power.values.reshape(-1, 1)
test_mains_reshaped = test_mains_power.values.reshape(-1, 1)

# Prepare data generators using TimeSeriesGenerator from Keras.
# This generator takes the time series data and creates batches for training.
# Window size (30): We use 30 time steps (assuming data resolution is 1 second).
# This helps capture short-term dependencies.
window_size = 150  # 600 (data points per hour), so 150 is equivalent to 15 minutes
batch_size = 32  # Batch size of 32 to increase training efficiency.

# The generator feeds the reshaped data into the model for training.
train_generator = TimeseriesGenerator(train_mains_reshaped, train_mains_reshaped,
                                      length=window_size, batch_size=batch_size)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped,
                                     length=window_size, batch_size=batch_size)


# Define a transformer encoder block for the model.
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # Multi-head Self Attention layer
    # head_size: Each attention head learns a different representation. Here it's set to 32.
    # num_heads: 2 attention heads allow the model to focus on different aspects of the input.
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)

    # Add & Normalize: Skip connection (Add) and LayerNormalization are common practices in transformers
    # to help the gradient flow and stabilize training.
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed Forward Network: This is a two-layer fully connected network with ReLU activation.
    # ff_dim: The dimension of the feed-forward network is 64, providing more capacity to learn complex patterns.
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dropout(dropout)(ff_output)  # Dropout is applied to prevent overfitting.

    # The output is then reduced back to the input size.
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

    return ff_output


# Define the Transformer model for NILM.
def create_transformer_model(shape_of_input, head_size=32, num_heads=2, ff_dim=64, num_transformer_blocks=2,
                             dropout=0.2):
    # The input is a time series with a shape defined by `window_size` and 1 feature (the power value).
    inputs = Input(shape=shape_of_input)

    x = inputs
    # Add several transformer blocks. Each block uses the transformer_encoder function.
    # num_transformer_blocks: We stack 2 transformer blocks, each learning more complex representations.
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # The output is flattened before being passed to the final Dense layer.
    # Flattening allows us to reduce the multidimensional output of the transformers into a 1D vector.
    x = Flatten()(x)

    # The final Dense layer outputs a single value (since NILM is a regression problem, we use 'linear' activation).
    outputs = Dense(1, activation='linear')(x)

    # Create and compile the model using Adam optimizer.
    # Adam is chosen for its adaptive learning rate capabilities, commonly used in deep learning.
    # Loss: Mean Squared Error (MSE) is typical for regression tasks.
    model = Model(inputs, outputs)

    return model


# Create and train the Transformer model.
# The input shape is defined by the `window_size` (30 time steps) and 1 feature (power consumption).
input_shape = (window_size, 1)
transformer_model = create_transformer_model(input_shape)

# compile the model using wandb config
transformer_model.compile(optimizer=config.optimizer,
                          loss=config.loss,
                          metrics=[config.metric]
                          )

# Print model summary to view the structure.
transformer_model.summary()

# Train the model.
# We train the model for n epochs (wandb based), validating on the test data.
# The generators automatically provide data in the correct format for training and validation.
# WandbMetricsLogger will log train and validation metrics to wandb
# WandbModelCheckpoint will upload model checkpoints to wandb
history = transformer_model.fit(train_generator,
                                epochs=config.epoch,
                                batch_size=config.batch_size,
                                validation_data=test_generator,
                                callbacks=[
                                    WandbMetricsLogger(log_freq=5),
                                    WandbModelCheckpoint("models")
                                ])

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
import numpy as np
from time_series_manager import TimeSeries
from tensorflow.keras.mixed_precision import set_global_policy, Policy
from tensorflow.keras import backend as tfk_backend

tfk_backend.clear_session()
policy = Policy('mixed_float16')
set_global_policy(policy)

# Set GPU memory growth to avoid GPU memory issues
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass  # Invalid device or cannot modify virtual devices once initialized


# Function to create a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Function to create a Seq2Point model
def create_seq2point_model(input_shape_param, units=64):
    # Input layer
    inputs = Input(shape=input_shape_param, name="input_layer")

    # Convolutional layers: two. For no reason. Check others' implementation. Fundaments, etc
    conv1 = Conv1D(filters=units, kernel_size=3, activation='relu', padding='same')(inputs)
    conv2 = Conv1D(filters=units, kernel_size=3, activation='relu', padding='same')(conv1)

    # Flatten the output of the last convolutional layer
    flatten = Flatten()(conv2)

    # Fully connected layer to output a single value (appliance power consumption)
    dense = Dense(units=units, activation='relu')(flatten)
    # TODO: Where is the dropout?
    # TODO: What is the dimensionality of the output since it's a "point"
    outputs = Dense(1, activation='linear', dtype=tf.float32)(dense)

    # Define the model
    model_aux = Model(inputs=inputs, outputs=outputs)
    model_aux.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

    return model_aux


# Function to create input/output sequences from the data (for Seq2Point)
# TODO: Go deeper here. Study to understand why.
def create_seq2point_sequences(df, input_seq_length):

#TODO: Check a well stablished article + impl regarding handoing the input sequence
#(time_step, total_power_consumption, appliance, appliance_consumption)xn =>
#(time_step, appliance, appliance_consumption)

    input_sequences = []
    target_values = []
    for i in range(len(df) - input_seq_length):
        input_seq = df['power'].values[i:i + input_seq_length]
        target_value = df['power'].values[i + input_seq_length]  # Target is the next step value
        input_sequences.append(input_seq)
        target_values.append(target_value)
    return np.array(input_sequences), np.array(target_values)


# Normalize the data
def normalize_data(train_data, test_data):
    mean = train_data['power'].mean()
    std = train_data['power'].std()
    train_data['power'] = (train_data['power'] - mean) / std
    test_data['power'] = (test_data['power'] - mean) / std
    return train_data, test_data


print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(physical_devices))

# Parameters
input_sequence_length = 60  # The length of the input sequence
input_shape = (input_sequence_length, 1)  # Shape of the input sequences

# Load and preprocess training data (Building 1)
train_df = load_data(filepath='../datasets/ukdale.h5', building=1, start_time='2014-01-01', end_time='2015-02-15')

# Load and preprocess testing data (Building 5)
test_df = load_data(filepath='../datasets/ukdale.h5', building=5, start_time='2014-01-01', end_time='2015-02-15')

# Normalize data
train_df, test_df = normalize_data(train_df, test_df)

# Create input sequences and target values for training data
train_input_sequences, train_target_values = create_seq2point_sequences(train_df, input_sequence_length)

# Create input sequences and target values for testing data
test_input_sequences, test_target_values = create_seq2point_sequences(test_df, input_sequence_length)

# Reshape data to be compatible with the model
train_input_sequences = np.expand_dims(train_input_sequences, axis=-1)  # Shape: (num_samples, input_sequence_length, 1)
test_input_sequences = np.expand_dims(test_input_sequences, axis=-1)  # Shape: (num_samples, input_sequence_length, 1)

# Prepare the data as TensorFlow datasets
train_seq2point_dataset = tf.data.Dataset.from_tensor_slices((train_input_sequences, train_target_values)).batch(4)
test_seq2point_dataset = tf.data.Dataset.from_tensor_slices((test_input_sequences, test_target_values)).batch(4)

# Create the Seq2Point model
model = create_seq2point_model(input_shape, units=32)

# Print model summary
model.summary()

# Callbacks for learning rate scheduling, early stopping, and model checkpointing
callbacks = [
    LearningRateScheduler(scheduler),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('../models/seq2point_model.keras', save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir='../logs')
]

# Fit the model using the training data and validate on testing data
model.fit(train_seq2point_dataset, epochs=10, validation_data=test_seq2point_dataset, callbacks=callbacks)

# Evaluate the model on the testing data
test_loss, test_mae = model.evaluate(test_seq2point_dataset)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

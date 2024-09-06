import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from load_data import load_data


# Function to create a simplified Seq2Seq model
def create_simple_seq2seq_model(input_shape_param, output_sequence_length, units=64):
    # Encoder
    encoder_inputs = Input(shape=input_shape_param, name="encoder_input")
    encoder_outputs, state_h, state_c = LSTM(units, return_state=True)(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(output_sequence_length, input_shape_param[1]), name="decoder_input")
    decoder_lstm = LSTM(units, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # TimeDistributed Layer to generate the output
    decoder_dense = TimeDistributed(Dense(1, activation='linear', dtype=tf.float32), name="output")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

    return model


# Function to create input/output sequences from the data
def create_sequences(df, input_sequence_length, output_sequence_length):
    input_sequences = []
    output_sequences = []
    for i in range(len(df) - input_sequence_length - output_sequence_length):
        input_seq = df['power'].values[i:i + input_sequence_length]
        output_seq = df['power'].values[i + input_sequence_length:i + input_sequence_length + output_sequence_length]
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    return np.array(input_sequences), np.array(output_sequences)


print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Parameters
input_sequence_length = 60
output_sequence_length = 30
input_shape = (input_sequence_length, 1)  # Shape of the input sequences

# Load and preprocess training data (Building 1)
train_df = load_data(filepath='ukdale.h5', building=1, start_time='2014-01-01', end_time='2015-02-15')

# Load and preprocess testing data (Building 5)
test_df = load_data(filepath='ukdale.h5', building=5, start_time='2014-01-01', end_time='2015-02-15')

# Create input and output sequences for training data
train_input_sequences, train_output_sequences = create_sequences(train_df, input_sequence_length,
                                                                 output_sequence_length)

# Create input and output sequences for testing data
test_input_sequences, test_output_sequences = create_sequences(test_df, input_sequence_length, output_sequence_length)

# Reshape data to be compatible with the model
train_input_sequences = np.expand_dims(train_input_sequences, axis=-1)  # Shape: (num_samples, input_sequence_length, 1)
train_output_sequences = np.expand_dims(train_output_sequences,
                                        axis=-1)  # Shape: (num_samples, output_sequence_length, 1)

test_input_sequences = np.expand_dims(test_input_sequences, axis=-1)  # Shape: (num_samples, input_sequence_length, 1)
test_output_sequences = np.expand_dims(test_output_sequences,
                                       axis=-1)  # Shape: (num_samples, output_sequence_length, 1)

# Prepare the data as TensorFlow datasets
train_seq2seq_dataset = tf.data.Dataset.from_tensor_slices(
    ((train_input_sequences, np.zeros_like(train_output_sequences)), train_output_sequences)
).batch(16)

test_seq2seq_dataset = tf.data.Dataset.from_tensor_slices(
    ((test_input_sequences, np.zeros_like(test_output_sequences)), test_output_sequences)
).batch(16)

# Create the Seq2Seq model
model = create_simple_seq2seq_model(input_shape, output_sequence_length, units=64)

# Print model summary
model.summary()

# Fit the model using the training data and validate on testing data
model.fit(train_seq2seq_dataset, epochs=10, validation_data=test_seq2seq_dataset)

# Evaluate the model on the testing data
test_loss, test_mae = model.evaluate(test_seq2seq_dataset)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

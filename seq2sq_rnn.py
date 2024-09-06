import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


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


# Parameters
input_sequence_length = 60
output_sequence_length = 30
input_shape = (input_sequence_length, 1)  # Shape of the input sequences

# Create the model
model = create_simple_seq2seq_model(input_shape, output_sequence_length, units=64)

# Print model summary
model.summary()

# Generate dummy data for demonstration
num_samples = 100
encoder_input_data = np.random.rand(num_samples, input_sequence_length, 1).astype(np.float32)
decoder_input_data = np.random.rand(num_samples, output_sequence_length, 1).astype(np.float32)
decoder_target_data = np.random.rand(num_samples, output_sequence_length, 1).astype(np.float32)

# Prepare the data as a dataset
train_seq2seq_dataset = tf.data.Dataset.from_tensor_slices(
    ((encoder_input_data, decoder_input_data), decoder_target_data)
).batch(16)

# Fit the model
model.fit(train_seq2seq_dataset, epochs=10, validation_data=train_seq2seq_dataset)

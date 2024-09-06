import pandas as pd
import hdf5plugin
import h5py
import numpy as np
import tensorflow as tf
import load_data as ld
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Input, Flatten, RepeatVector, TimeDistributed
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam


def create_seq2seq_model(input_shape_param, output_sequence_length, units=64):
    # Encoder
    encoder_inputs = Input(shape=input_shape_param)
    encoder = LSTM(units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(output_sequence_length, input_shape_param[1]))  # Expected output sequence length
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Dense layer for generating the output sequence
    decoder_dense = TimeDistributed(Dense(1, activation='linear'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Build the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

    return model


# Define parameters for the TimeseriesGenerator
window_size = 60  # Input sequence length
output_sequence_length = 30  # Output sequence length
batch_size = 32

train_mains = ld.load_data('ukdale.h5', 1, '2014-01-01', '2015-02-15')
test_mains = ld.load_data('ukdale.h5', 5, '2014-01-01', '2015-02-15')


# Reshape the data to have a third dimension
train_mains_reshaped = train_mains['power'].values.reshape(-1, 1)
test_mains_reshaped = test_mains['power'].values.reshape(-1, 1)

# Create data generators for training and testing
train_generator = TimeseriesGenerator(train_mains_reshaped, train_mains_reshaped,
                                      length=window_size + output_sequence_length,
                                      batch_size=batch_size)

test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped,
                                     length=window_size + output_sequence_length,
                                     batch_size=batch_size)


# Function to generate inputs and outputs for Seq2Seq
def generate_seq2seq_data(generator, window_size, output_sequence_length):
    for batch_x, batch_y in generator:
        encoder_input = batch_x[:, :window_size, :]  # First part of the sequence
        decoder_input = batch_x[:, window_size:, :]  # Second part as decoder input
        decoder_output = batch_y[:, window_size:, :]  # Second part as the expected output sequence
        yield [encoder_input, decoder_input], decoder_output


# Create generators for Seq2Seq
train_seq2seq_generator = generate_seq2seq_data(train_generator, window_size, output_sequence_length)
test_seq2seq_generator = generate_seq2seq_data(test_generator, window_size, output_sequence_length)

# Define the input shape for the model
input_shape = (window_size, 1)

# Create the Seq2Seq model
model = create_seq2seq_model(input_shape, output_sequence_length, units=64)

# Train the model
model.fit(train_seq2seq_generator, epochs=10, validation_data=test_seq2seq_generator,
          steps_per_epoch=len(train_generator), validation_steps=len(test_generator))

# Evaluate the model on the test data
loss, mae = model.evaluate(test_seq2seq_generator, steps=len(test_generator))
print(f'Test MAE: {mae}')

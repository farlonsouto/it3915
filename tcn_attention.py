import pandas as pd
import hdf5plugin
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Flatten, Multiply, Lambda, Add
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Load and pre-process the data
def load_data(filepath, building, start_time, end_time):
    with h5py.File(filepath, 'r') as f:
        mains_data = f[f'building{building}/elec/meter1/table']
        index = mains_data['index'][:]
        values = mains_data['values_block_0'][:]
        timestamps = pd.to_datetime(index)
        power = values.flatten()
        df = pd.DataFrame({'timestamp': timestamps, 'power': power})
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        df.set_index('timestamp', inplace=True)
        return df


train_mains = load_data('ukdale.h5', 1, '2014-01-01', '2015-02-15')
test_mains = load_data('ukdale.h5', 5, '2014-01-01', '2015-02-15')
max_power = train_mains['power'].max()
train_mains['power'] = train_mains['power'] / max_power
test_mains['power'] = test_mains['power'] / max_power


# Define the Attention and TCN models

def attention_block(inputs):
    """
    Attention mechanism for Temporal Convolutional Network. This part of the code introduces an attention mechanism that
    helps the model to focus on the relevant parts of the input sequence.
    """
    attention = Dense(1, activation='tanh')(inputs)
    attention = Flatten()(attention)
    attention = Dense(inputs.shape[1], activation='softmax')(attention)
    attention = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention)
    return Multiply()([inputs, attention])


def temporal_block(x, dilation_rate, nb_filters, kernel_size, padding='causal', activation='relu'):
    """
    Temporal block for Temporal Convolutional Network. Defines the core structure of the TCN with causal, dilated
    convolutions to handle long-term dependencies.
    """
    conv = Conv1D(nb_filters, kernel_size, padding=padding, dilation_rate=dilation_rate, activation=activation)(x)
    conv = Conv1D(nb_filters, kernel_size, padding=padding, dilation_rate=dilation_rate, activation=activation)(conv)
    return conv


def create_tcn_model(input_shape_param, nb_filters=16, kernel_size=4, nb_stacks=3):
    """
    Create the Temporal Convolutional Network with Attention mechanism. It integrates both the temporal blocks and the
    attention mechanisms to form the full model.
    """
    inputs = Input(shape=input_shape_param)
    x = inputs

    for i in range(nb_stacks):
        x = temporal_block(x, dilation_rate=2 ** i, nb_filters=nb_filters, kernel_size=kernel_size)
        x = attention_block(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)

    tcn_model = Model(inputs, outputs)
    tcn_model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return tcn_model


# Prepare data generators

# TODO adjust the hyperparameters: nb_filters, kernel_size, and the number of temporal blocks (nb_stacks), based on the
#  specific requirements intrinsic to NILM and on the computational resources available (e.g: local machine, one GPU).

window_size = 60
batch_size = 32

train_mains_reshaped = train_mains['power'].values.reshape(-1, 1)
test_mains_reshaped = test_mains['power'].values.reshape(-1, 1)

train_generator = TimeseriesGenerator(train_mains_reshaped, train_mains_reshaped,
                                      length=window_size, batch_size=batch_size)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped,
                                     length=window_size, batch_size=batch_size)

# Create and train the model

input_shape = (window_size, 1)
model = create_tcn_model(input_shape)

# The following line trains the model for 10 epochs. Each epoch consists of training the model on a batch of data. And
# each batch of data is made up of 32 samples.
model.fit(train_generator, epochs=10, validation_data=test_generator)

loss, mae = model.evaluate(test_generator)
print(f'Test MAE: {mae}')

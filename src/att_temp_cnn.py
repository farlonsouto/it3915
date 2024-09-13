import tensorflow as tf
import helper as ld
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Flatten, Multiply, Lambda
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Multiply, Reshape

# Load data
train_mains = ld.load_data('../datasets/ukdale.h5', 1, '2014-01-01', '2015-02-15')
test_mains = ld.load_data('../datasets/ukdale.h5', 5, '2014-01-01', '2015-02-15')

# Normalize power values
max_power = train_mains['power'].max()
train_mains['power'] = train_mains['power'] / max_power
test_mains['power'] = test_mains['power'] / max_power


# Define the Attention and TCN models


def attention_block(inputs):
    # Step 1: Calculate attention scores with a Dense layer and tanh activation
    attention = Dense(1, activation='tanh')(inputs)

    # Step 2: Flatten the attention scores to (batch_size, sequence_length)
    attention = Flatten()(attention)

    # Step 3: Apply a Dense layer with softmax activation to get attention weights
    attention = Dense(inputs.shape[1], activation='softmax')(attention)

    # Step 4: Reshape attention weights to (batch_size, sequence_length, 1)
    attention = Reshape((inputs.shape[1], 1))(attention)  # Then, no Lambda

    # Step 5: Multiply the inputs by the attention weights
    output = Multiply()([inputs, attention])

    return output


def temporal_block(x, dilation_rate, nb_filters, kernel_size, padding='causal', activation='relu'):
    conv = Conv1D(nb_filters, kernel_size, padding=padding, dilation_rate=dilation_rate, activation=activation)(x)
    conv = Conv1D(nb_filters, kernel_size, padding=padding, dilation_rate=dilation_rate, activation=activation)(conv)
    return conv


def create_tcn_model(input_shape_param, nb_filters=16, kernel_size=4, nb_stacks=3):
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
window_size = 15
batch_size = 8

train_mains_reshaped = train_mains['power'].values.reshape(-1, 1)
test_mains_reshaped = test_mains['power'].values.reshape(-1, 1)

train_generator = TimeseriesGenerator(train_mains_reshaped, train_mains_reshaped, length=window_size,
                                      batch_size=batch_size)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped, length=window_size,
                                     batch_size=batch_size)

# Create and compile the model
input_shape = (window_size, 1)
model = create_tcn_model(input_shape)

model.summary()


# Define a simple learning rate scheduler
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * 0.5
    return lr


# Define callbacks for learning rate scheduling, early stopping, model checkpointing, and TensorBoard logging
callbacks = [
    LearningRateScheduler(scheduler),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    #   ModelCheckpoint('../models/latest_att_temp_cnn.keras', save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir='../logs')
]

# Train the model with callbacks
model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=callbacks)

model.save('../models/latest_att_temp_cnn.keras')

# Evaluate the model
loss, mae = model.evaluate(test_generator)
print(f'Test MAE: {mae}')

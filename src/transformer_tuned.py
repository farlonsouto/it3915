import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load data using NILMTK
# Training data: Building 1, from '2014-01-01' to '2015-02-15'
dataset = DataSet('../datasets/ukdale.h5')
dataset.set_window(start='2014-01-01', end='2015-02-15')

train_elec = dataset.buildings[1].elec
train_mains = train_elec.mains()

test_elec = dataset.buildings[5].elec
test_mains = test_elec.mains()

# Load the data into DataFrames
train_mains_df = next(train_mains.load())
test_mains_df = next(test_mains.load())

# Flatten MultiIndex columns if necessary
if isinstance(train_mains_df.columns, pd.MultiIndex):
    train_mains_df.columns = ['_'.join(col).strip() for col in train_mains_df.columns.values]
if isinstance(test_mains_df.columns, pd.MultiIndex):
    test_mains_df.columns = ['_'.join(col).strip() for col in test_mains_df.columns.values]

# Access the 'power_active' column
train_mains_power = train_mains_df['power_active']
test_mains_power = test_mains_df['power_active']

# Normalize data
max_power = train_mains_power.max()
train_mains_power = train_mains_power / max_power
test_mains_power = test_mains_power / max_power

# Reshape data
train_mains_reshaped = train_mains_power.values.reshape(-1, 1)
test_mains_reshaped = test_mains_power.values.reshape(-1, 1)

# Prepare data generators
window_size = 60
batch_size = 32

train_generator = TimeseriesGenerator(train_mains_reshaped, train_mains_reshaped,
                                      length=window_size, batch_size=batch_size)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped,
                                     length=window_size, batch_size=batch_size)


# Define Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # Multi-head Self Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed Forward Network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

    return ff_output


# Define the Transformer model for NILM
def create_transformer_model(shape_of_input, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=4,
                             dropout=0.1):
    inputs = Input(shape=shape_of_input)

    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Final output layer (since this is a regression task, we use linear activation)
    x = Flatten()(x)
    outputs = Dense(1, activation='linear')(x)

    # Create model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

    return model


# Create and train the Transformer model
input_shape = (window_size, 1)
transformer_model = create_transformer_model(input_shape)

# Print model summary
transformer_model.summary()


# Define callbacks
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * 0.5
    return lr


callbacks = [
    LearningRateScheduler(scheduler),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('../models/transformer_model.keras', save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir='../logs')
]

# Train the model
transformer_model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=callbacks)

# Evaluate the model
loss, mae = transformer_model.evaluate(test_generator)
print(f'Test MAE: {mae}')

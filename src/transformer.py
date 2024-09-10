import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten, Add
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
import helper as ld
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard

# Load and preprocess data
train_mains = ld.load_data('../datasets/ukdale.h5', 1, '2014-01-01', '2015-02-15')
test_mains = ld.load_data('../datasets/ukdale.h5', 5, '2014-01-01', '2015-02-15')

# Normalize data
max_power = train_mains['power'].max()
train_mains['power'] = train_mains['power'] / max_power
test_mains['power'] = test_mains['power'] / max_power


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


# Prepare data generators
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

window_size = 60
batch_size = 32

train_mains_reshaped = train_mains['power'].values.reshape(-1, 1)
test_mains_reshaped = test_mains['power'].values.reshape(-1, 1)

train_generator = TimeseriesGenerator(train_mains_reshaped, train_mains_reshaped,
                                      length=window_size, batch_size=batch_size)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped,
                                     length=window_size, batch_size=batch_size)

# Create and train the Transformer model
input_shape = (window_size, 1)
transformer_model = create_transformer_model(input_shape)

# Print model summary
transformer_model.summary()


# Define a simple learning rate scheduler
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * 0.5
    return lr


# Define callbacks for learning rate scheduling, early stopping, model checkpointing, and TensorBoard logging
callbacks = [
    LearningRateScheduler(scheduler),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('../models/transformer_model.keras', save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir='../logs')
]

# Train the model
transformer_model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
loss, mae = transformer_model.evaluate(test_generator)
print(f'Test MAE: {mae}')

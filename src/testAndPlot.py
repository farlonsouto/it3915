import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import helper as ld
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


# Custom function for the Lambda layer operation (expand dimensions)
def custom_expand_dims(x):
    return tf.expand_dims(x, axis=-1)


# Custom output shape function for the Lambda layer
def custom_lambda_output_shape(input_shape):
    # Add an extra dimension at the end
    return input_shape + (1,)


# Custom function to create a Lambda layer with the correct output shape
def create_lambda_layer():
    return tf.keras.layers.Lambda(custom_expand_dims, output_shape=custom_lambda_output_shape)


# Enable unsafe deserialization (for compatibility with older models)
tf.keras.config.enable_unsafe_deserialization()

# Load the model with the custom Lambda functions
model = load_model(
    '../models/att_temp_cnn_vanilla.keras',
    custom_objects={
        'custom_expand_dims': custom_expand_dims,
        'custom_lambda_output_shape': custom_lambda_output_shape,
        'Lambda': create_lambda_layer  # Explicitly provide Lambda layer with custom functions
    }
)
# Load the test data for building 5
test_mains = ld.load_data('ukdale.h5', 5, '2014-01-01', '2015-02-15')

# Normalize the test data
max_power = test_mains['power'].max()
test_mains['power'] = test_mains['power'] / max_power

# Prepare the test data generator
window_size = 60
batch_size = 32

test_mains_reshaped = test_mains['power'].values.reshape(-1, 1)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped, length=window_size,
                                     batch_size=batch_size)

# Generate predictions using the pre-trained model
predictions = model.predict(test_generator)

# Rescale predictions back to the original power scale
predictions_rescaled = predictions * max_power

# Rescale the test data back to the original power scale
actual_rescaled = test_mains['power'].values[window_size:] * max_power

# Prepare the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel('Time Steps')
ax.set_ylabel('Power Consumption (W)')
ax.set_title('Disaggregated Appliance Power Consumption')

# Initialize the lines for each appliance
lines = [ax.plot([], [], label=f'Appliance {i + 1}')[0] for i in range(predictions_rescaled.shape[1])]

# Set plot limits
ax.set_xlim(0, len(predictions_rescaled))
ax.set_ylim(0, max(predictions_rescaled.max(), actual_rescaled.max()))


# Animation function to update the plot
def animate(i):
    for idx, line in enumerate(lines):
        line.set_data(range(i), predictions_rescaled[:i, idx])
    return lines


# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(predictions_rescaled), interval=100, blit=True)

# Add legend
ax.legend()

# Show the plot with animation
plt.show()

print("end of processing.")

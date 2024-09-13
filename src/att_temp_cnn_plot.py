import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import keras

import helper as ld


# Load model and pass custom_objects that ensure 'tf' is included
model = keras.models.load_model('../models/latest_att_temp_cnn.keras', compile=True, safe_mode=True)

model.summary()

# Load and preprocess test data
test_mains = ld.load_data('../datasets/ukdale.h5', 5, '2014-01-01', '2015-09-01')
max_power = test_mains['power'].max()
test_mains['power'] = test_mains['power'] / max_power

window_size = 15
batch_size = 32

test_mains_reshaped = test_mains['power'].values.reshape(-1, 1)
test_generator = TimeseriesGenerator(test_mains_reshaped, test_mains_reshaped, length=window_size,
                                     batch_size=batch_size)

# Generate predictions
predictions = model.predict(test_generator)

# Rescale predictions and actual values
predictions_rescaled = predictions * max_power
actual_rescaled = test_mains['power'].values[window_size:] * max_power

# Prepare plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel('Time Steps')
ax.set_ylabel('Power Consumption (W)')
ax.set_title('Disaggregated Appliance Power Consumption')

lines = [ax.plot([], [], label='Predicted Power')[0]]

ax.set_xlim(0, len(predictions_rescaled))
ax.set_ylim(0, max(predictions_rescaled.max(), actual_rescaled.max()))


# Animation function
def animate(i):
    lines[0].set_data(range(i), predictions_rescaled[:i])
    return lines


ani = animation.FuncAnimation(fig, animate, frames=len(predictions_rescaled), interval=100, blit=True)

ax.legend()
plt.show()

print("end of processing.")

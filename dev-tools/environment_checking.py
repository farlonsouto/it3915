import tensorflow as tf

# Check if TensorFlow detects the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.sysconfig.get_build_info())

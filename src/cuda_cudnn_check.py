import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info

print(tf_build_info.build_info)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

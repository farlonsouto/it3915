import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info


print("Tensorflow version: ",tf.__version__)
print("Keras version: ", tf.keras.__version__)


print(tf_build_info.build_info)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

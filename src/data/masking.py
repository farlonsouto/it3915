import tensorflow as tf
from keras import layers


class Mask(layers.Layer):
    """ The Masked Learning Model mask abstraction. A mask marks the portion of the input that is to
    be used in the loss computation. The original value should be replaced by something else"""

    def __init__(self, config):
        super(Mask, self).__init__()
        self.mlm_mask = config.mlm_mask
        self.config = config
        self.mask_token = config.mask_token  # Assuming this is defined in the config
        self.masking_portion = config.masking_portion

    def build(self, input_shape):
        super().build(input_shape)
        self.input_dim = input_shape[-1]

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        windows_size = tf.shape(inputs)[1]

        masked_inputs = inputs
        if self.mlm_mask and training:
            # Create masking tensor
            mask = tf.random.uniform((batch_size, windows_size)) < self.masking_portion
            # Expand mask to match input dimensions
            mask = tf.expand_dims(mask, -1)
            mask = tf.tile(mask, [1, 1, self.input_dim])

            # Create masked version of input
            masked_values = tf.ones_like(inputs) * self.mask_token
            masked_inputs = tf.where(mask, masked_values, inputs)

        return masked_inputs

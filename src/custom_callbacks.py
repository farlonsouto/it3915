import tensorflow as tf


class GradientDebugCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        with tf.GradientTape() as tape:
            # Forward pass on a batch
            predictions = self.model(self.validation_data[0], training=True)
            loss = self.model.compiled_loss(self.validation_data[1], predictions)

        # Calculate gradients for trainable variables
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Check for NaNs or extreme values
        if any(tf.reduce_any(tf.math.is_nan(g)) for g in gradients):
            print(f"NaN gradients detected in batch {batch}")
        else:
            print(f"Gradients for batch {batch} are within normal range.")


class BatchStatsCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        # Print the batch loss from logs
        if 'loss' in logs:
            print(f"Batch {batch} - Loss: {logs['loss']}")

        # Compute gradients using GradientTape within the callback
        with tf.GradientTape() as tape:
            # Re-run the forward pass to compute gradients
            predictions = self.model(self.model.train_function.inputs[0], training=True)
            loss = self.model.compiled_loss(
                self.model.train_function.inputs[1], predictions, regularization_losses=self.model.losses)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Check for NaNs in the gradients
        for i, grad in enumerate(gradients):
            if tf.reduce_any(tf.math.is_nan(grad)):
                print(f"NaN gradient detected in batch {batch} for gradient {i}")

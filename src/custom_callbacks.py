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
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}

        # Print batch loss
        if 'loss' in logs:
            print(f"Batch {batch} - Loss: {logs['loss']}")

        try:
            # Access the inputs and labels of the current batch within a GradientTape
            with tf.GradientTape() as tape:
                inputs, labels = self.model._current_inputs, self.model._current_labels
                predictions = self.model(inputs, training=True)  # Forward pass
                loss = self.model.compiled_loss(labels, predictions, regularization_losses=self.model.losses)

            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # Check for NaNs in gradients
            for i, grad in enumerate(gradients):
                if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
                    print(f"NaN gradient detected in batch {batch} for gradient {i}")

        except Exception as e:
            print(f"Error in BatchStatsCallback on batch {batch}: {e}")




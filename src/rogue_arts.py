import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class HarryPlotter:
    def __init__(self, model, test_data_generator):
        """
        Initializes the Plotter class.

        Args:
            model (tf.keras.Model): The trained model for inference.
            test_data_generator: A generator that provides test data (inputs and ground truth).
        """
        self.model = model
        self.test_data_generator = test_data_generator
        self.fig, self.ax = plt.subplots()
        self.line_gt, = self.ax.plot([], [], label="Ground Truth", color="blue")
        self.line_pred, = self.ax.plot([], [], label="Prediction", color="red")
        self.ax.legend()
        self.ax.set_title("Real-Time Predictions vs. Ground Truth")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Value")
        self.ground_truth = []
        self.predictions = []

    def update_plot(self, x, y):
        """
        Updates the plot with new predictions and ground truth values.

        Args:
            x: Input batch from the test data generator.
            y: Ground truth batch corresponding to `x`.
        """
        # Perform prediction on the current batch
        pred = self.model.predict(x)

        # Append the new data to the lists
        self.ground_truth.extend(y.flatten())
        self.predictions.extend(pred.flatten())

        # Update the plot data
        self.line_gt.set_data(range(len(self.ground_truth)), self.ground_truth)
        self.line_pred.set_data(range(len(self.predictions)), self.predictions)

        # Adjust the plot limits dynamically
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the plot
        plt.pause(0.01)

    def run(self):
        """
        Executes the real-time plotting during inference.
        """
        print("Starting real-time plotting...")
        for x, y in self.test_data_generator:
            self.update_plot(x, y)

        print("Inference completed. Close the plot window to exit.")
        plt.show()

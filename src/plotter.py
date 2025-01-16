import matplotlib.pyplot as plt
import numpy as np


# Plotting function
def plot_comparison(test_gen, bert_model, title='Energy Consumption: Predicted vs Ground Truth'):
    # TODO: Plot the entire test data
    try:
        X_test, y_true, mask = next(iter(test_gen))  # Get the first batch of test data
    except ValueError:
        X_test, y_true = next(iter(test_gen))  # Get the first batch of test data
    y_pred = bert_model.predict(X_test)

    samples = y_true.shape[0] * y_true.shape[1]
    x = np.arange(samples)

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    plt.figure(figsize=(20, 10))
    plt.plot(x, y_true_flat, label='Ground Truth', color='blue', linewidth=2, alpha=0.7)
    plt.plot(x, y_pred_flat, label='Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)

    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Power (W)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)

    y_min = min(np.min(y_true_flat), np.min(y_pred_flat))
    y_max = max(np.max(y_true_flat), np.max(y_pred_flat))
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()
    plt.show()

    # Plot the difference
    plt.figure(figsize=(20, 10))
    plt.plot(x, y_pred_flat - y_true_flat, label='Prediction Error', color='green', linewidth=2)
    plt.title('Prediction Error', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Error (Predicted - Ground Truth)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

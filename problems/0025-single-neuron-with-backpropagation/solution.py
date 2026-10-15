
import numpy as np

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[np.ndarray, float, list[float]]:

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    weights = np.array(initial_weights, dtype=float).copy()
    bias = float(initial_bias)

    n = len(y)
    mse_values = []

    for _ in range(epochs):
        # 1. Forward pass for the entire batch
        z = X @ weights + bias
        predictions = 1 / (1 + np.exp(-z))

        # 2. Record MSE before updating parameters
        errors = predictions - y
        mse = np.mean(errors ** 2)
        mse_values.append(mse)

        # 3. Backpropagation through sigmoid + MSE
        delta = 2 * errors * predictions * (1 - predictions)

        # 4. Average gradients over the batch
        dw = (X.T @ delta) / n
        db = np.mean(delta)

        # 5. One update per epoch
        weights -= learning_rate * dw
        bias -= learning_rate * db

    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)
    mse_values = [round(value, 4) for value in mse_values]

    return updated_weights, updated_bias, mse_values

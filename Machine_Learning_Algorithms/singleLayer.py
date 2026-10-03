# Sami Ullah
# Roll no: 22013122-022

import numpy as np

inputs = np.array([
    [0, 1],
    [2, 2],
    [1, 4]
])
output = np.array([[0], [1], [1]])
weights = np.array([[0.5], [0.4]])  # Reshaped to (2,1) for matrix math
b = 4
learning_rate = 0.1
epochs = 1000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train(w, b, x, y, lr, epochs):
    for epoch in range(epochs):
        # --- Forward Pass ---
        # Z = Xw + b (3,2) dot (2,1) -> (3,1)
        z = np.dot(x, w) + b
        predicted_output = sigmoid(z) # Activation
        error = y - predicted_output # Backpropagation
        # d_cost/d_pred * d_pred/d_z = error * sigmoid_derivative
        d_predicted_output = error * sigmoid_derivative(z)
        # w = w + dot(x.T, error_term) * lr
        w += np.dot(x.T, d_predicted_output) * lr # Update Weights and Bias
        b += np.sum(d_predicted_output) * lr

        if epoch % 100 == 0:
            loss = np.mean(np.square(error))
            print(f"Epoch {epoch} Loss: {loss:.4f}")

    return w, b


# training
final_weights, final_b = train(weights, b, inputs, output, learning_rate, epochs)

print("\n--- Final Results ---")
print("Weights:\n", final_weights)
print("Bias:", final_b)
print("New Predictions:\n", sigmoid(np.dot(inputs, final_weights) + final_b))

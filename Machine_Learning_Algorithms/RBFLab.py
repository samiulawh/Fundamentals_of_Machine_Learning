# Sami Ullah
# Roll no: 22013122-022

import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([
    [3, 1],
    [2, 1],
    [6, 2]
])

targets = np.array([1, 0, 1])

# RBF Layer (Hidden Layer)
def gaussian_rbf(x, c, sigma=1.0):
    distance = np.linalg.norm(x - c)
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))

# W3 input points as the centers for 3 hidden neurons
centers = inputs
sigma = 3.0

# Shape 3 samples, 3 hidden neurons
n_samples = inputs.shape[0]
n_hidden = centers.shape[0]
rbf_activations = np.zeros((n_samples, n_hidden))

for i in range(n_samples):
    for j in range(n_hidden):
        rbf_activations[i, j] = gaussian_rbf(inputs[i], centers[j], sigma)

# 3. Output Layer (Logistic/Sigmoid) setup
weights = np.random.random(n_hidden)  # 3 weights for 3 RBF neurons
learning_rate = 0.1
epochs = 1000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training Loop
for epoch in range(epochs):
    # Forward Pass
    # Linear combination: z = w1*h1 + w2*h2 + w3*h3
    z = np.dot(rbf_activations, weights)
    predictions = sigmoid(z)

    # Loss (MSE)
    loss = np.mean(0.5 * (targets - predictions) ** 2)

    # --- Backpropagation ---
    # Gradient of Loss w.r.t Predicted: -(Target - Prediction)
    error = predictions - targets
    # Gradient of Sigmoid: Prediction * (1 - Prediction)
    d_sigmoid = error * sigmoid_derivative(z)

    # Gradient of Weights: d_sigmoid * Input to the weights (rbf_activations)
    gradients = np.dot(rbf_activations.T, d_sigmoid)
    weights -= learning_rate * gradients

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal Predictions:")
print(np.round(predictions, 2))
print("Target Values:")
print(targets)

# plotting
plt.scatter(inputs[:, 0], inputs[:, 1])
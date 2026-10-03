# Sami Ullah
# Roll no: 22013122-022

import numpy as np

# Input(2) -> Hidden(3) -> Output(1)
inputs = np.array([[0.5, -0.2]])  # Shape (1, 2)
target = np.array([[1.0]])

np.random.seed(0)
# Layer 1 weights: 2 Inputs to 3 Hidden Neurons
w1 = np.random.randn(2, 3)
# print("w1: ", w1)
b1 = np.zeros((1, 3))
# Layer 2 weights: 3 Hidden Neurons to 1 Output
w2 = np.random.randn(3, 1)
b2 = np.zeros((1, 1))
learning_rate = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Training
for i in range(1000):
    # --- Forward Pass ---
    # Layer 1 (Hidden)
    # Dot product: (1,2) . (2,3) -> (1,3)
    z1 = np.dot(inputs, w1) + b1
    a1 = sigmoid(z1)  # Activation of hidden layer
    # Layer 2 (Output)
    # Dot product: (1,3) . (3,1) -> (1,1)
    z2 = np.dot(a1, w2) + b2
    output = sigmoid(z2)
    # --- Backpropagation ---
    error = output - target
    # Gradients for Output Layer (w2)
    # d_loss/d_output * d_output/d_z2
    delta_output = error * sigmoid_derivative(z2)
    w2_grad = np.dot(a1.T, delta_output) # Gradient w2
    b2_grad = np.sum(delta_output, axis=0, keepdims=True)

    # Gradients for Hidden Layer (w1)
    # (Delta Output) dot (w2.T) * d_hidden/d_z1
    error_hidden = np.dot(delta_output, w2.T)
    delta_hidden = error_hidden * sigmoid_derivative(z1)
    w1_grad = np.dot(inputs.T, delta_hidden) # Gradient w1
    b1_grad = np.sum(delta_hidden, axis=0, keepdims=True)

    w2 -= learning_rate * w2_grad
    b2 -= learning_rate * b2_grad
    w1 -= learning_rate * w1_grad
    b1 -= learning_rate * b1_grad

    if i % 100 == 0:
        print(f"Epoch {i}, Output: {output[0][0]:.4f}, Loss: {np.mean(error ** 2):.4f}")

print("\nFinal Output:")
print(output)
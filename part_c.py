import numpy as np
import matplotlib.pyplot as plt

# Create data
x = np.linspace(-10, 10, 1000).reshape(-1, 1)
y = 3 * x + 0.7 * x**2

# Define neural network
def relu(x):
    return np.maximum(0, x)

def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, w3) + b3
    return z3

# Initialize weights and biases
np.random.seed(42)
w1 = np.random.randn(1, 32)
b1 = np.zeros((1, 32))
w2 = np.random.randn(32, 32)
b2 = np.zeros((1, 32))
w3 = np.random.randn(32, 1)
b3 = np.zeros((1, 1))

# Train the model
learning_rate = 0.0001
epochs = 1000

for _ in range(epochs):
    # Forward pass
    y_pred = forward(x, w1, b1, w2, b2, w3, b3)
    
    # Compute loss
    loss = np.mean((y_pred - y)**2)
    
    # Backpropagation (simplified)
    grad = 2 * (y_pred - y) / len(x)
    w3 -= learning_rate * np.dot(forward(x, w1, b1, w2, b2, w3, b3).T, grad).T
    w2 -= learning_rate * np.dot(forward(x, w1, b1, w2, b2, w3, b3).T, grad).T
    w1 -= learning_rate * np.dot(x.T, grad)

# Test the model
x_test = np.linspace(-12, 12, 200).reshape(-1, 1)
y_test = 3 * x_test + 0.7 * x_test**2
y_pred = forward(x_test, w1, b1, w2, b2, w3, b3)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.1, label='Data')
plt.plot(x_test, y_test, color='green', label='True function')
plt.plot(x_test, y_pred, color='red', label='Neural Network')
plt.legend()
plt.title('Neural Network for y = 3x + 0.7x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Print error
mse = np.mean((y_test - y_pred)**2)
print(f"Mean Squared Error: {mse}")
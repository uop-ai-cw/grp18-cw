# pip install matplotlib
# pip install numpy
import numpy as np
import matplotlib.pyplot as plt

# Create data
x = np.linspace(-10, 10, 1000).reshape(-1, 1)
y = 3 * x + 0.7 * x**2

# Define neural network
def nn(x):
    return np.maximum(0, x)

def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(x, w1) + b1
    a1 = nn(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = nn(z2)
    z3 = np.dot(a2, w3) + b3
    return z3, z1, a1, z2, a2 

# Initialise weights and biases
np.random.seed(42)
w1 = np.random.randn(1, 32)
b1 = np.zeros((1, 32))
w2 = np.random.randn(32, 32)
b2 = np.zeros((1, 32))
w3 = np.random.randn(32, 1)
b3 = np.zeros((1, 1))

# Training the model
learning_rate = 0.0001
epochs = 2000

for epoch in range(epochs):
    y_pred, z1, a1, z2, a2 = forward(x, w1, b1, w2, b2, w3, b3)
    
    loss = np.mean((y_pred - y)**2)
    
    # Backpropagation
    grad_loss = 2 * (y_pred - y) / len(x) 
    
    # Gradients for Layer 3
    grad_w3 = np.dot(a2.T, grad_loss)
    grad_b3 = np.sum(grad_loss, axis=0, keepdims=True)
    grad_a2 = np.dot(grad_loss, w3.T)
    
    # Gradients for Layer 2
    grad_z2 = grad_a2 * (z2 > 0) 
    grad_w2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
    grad_a1 = np.dot(grad_z2, w2.T)
    
    # Gradients for Layer 1
    grad_z1 = grad_a1 * (z1 > 0) 
    grad_w1 = np.dot(x.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
    
    # Update weights and biases
    w3 -= learning_rate * grad_w3
    b3 -= learning_rate * grad_b3
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the model
x_test = np.linspace(-12, 12, 200).reshape(-1, 1)
y_test = 3 * x_test + 0.7 * x_test**2
y_pred, _, _, _, _ = forward(x_test, w1, b1, w2, b2, w3, b3)

# Print error
mse = np.mean((y_test - y_pred)**2)
print(f"Mean Squared Error: {mse}")

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
    

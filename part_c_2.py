import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # Forward propagation
        self.layer1 = self.sigmoid(np.dot(x, self.w1) + self.b1)
        self.output = np.dot(self.layer1, self.w2) + self.b2
        return self.output
    
    def train(self, x, y, learning_rate=0.01, epochs=1000):
        for _ in range(epochs):
            # Forward propagation
            layer1 = self.sigmoid(np.dot(x, self.w1) + self.b1)
            output = np.dot(layer1, self.w2) + self.b2
            
            # Backward propagation
            output_error = y - output
            output_delta = output_error
            
            layer1_error = np.dot(output_delta, self.w2.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(layer1)
            
            # Update weights and biases
            self.w2 += learning_rate * np.dot(layer1.T, output_delta)
            self.b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
            self.w1 += learning_rate * np.dot(x.T, layer1_delta)
            self.b1 += learning_rate * np.sum(layer1_delta, axis=0, keepdims=True)

# Generate training data
x = np.linspace(-2, 2, 100).reshape(-1, 1)
y = 3 * x + 0.7 * x**2

# Create and train the neural network
nn = SimpleNeuralNetwork(input_size=1, hidden_size=5, output_size=1)
nn.train(x, y, learning_rate=0.01, epochs=2000)

# Generate predictions
predictions = nn.forward(x)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Actual Function (y=3x+0.7xÂ²)', color='blue')
plt.plot(x, predictions, label='Neural Network Prediction', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Function Approximation')
plt.legend()
plt.grid(True)
plt.show()

# Test the network with some sample points
test_points = np.array([-1, 0, 1, 2]).reshape(-1, 1)
print("\nTest Points Comparison:")
print("x\t\tActual\t\tPredicted")
print("-" * 40)
for x_test in test_points:
    actual = float(3 * x_test + 0.7 * x_test**2)
    predicted = float(nn.forward(x_test.reshape(1, -1)))
    print(f"{x_test[0]:.1f}\t\t{actual:.3f}\t\t{predicted:.3f}")
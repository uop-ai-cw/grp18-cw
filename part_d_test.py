import numpy as np
import matplotlib.pyplot as plt
import csv

def normalise_data(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)

def denormalise_data(data, min_value, max_value):
    return data * (max_value - min_value) + min_value

def parse_csv(csv_path):
    training_inputs = []
    training_targets = []

    with open(csv_path, "r") as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]

    input_keys = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    cut_mapping = {"Fair": 1.0, "Good": 2.0, "Very Good": 3.0, "Premium": 4.0, "Ideal": 5.0}
    color_mapping = {"D": 7.0, "E": 6.0, "F": 5.0, "G": 4.0, "H": 3.0, "I": 2.0, "J": 1.0}
    clarity_mapping = {"I1": 1.0, "SI2": 2.0, "SI1": 3.0, "VS2": 4.0, "VS1": 5.0, "VVS2": 6.0, "VVS1": 7.0, "IF": 8.0}

    for element in data:
        element["carat"] = normalise_data(float(element["carat"]), 0.2, 5.01)
        element["cut"] = normalise_data(cut_mapping[element["cut"]], 1.0, 5.0)
        element["color"] = normalise_data(color_mapping[element["color"]], 1.0, 7.0)
        element["clarity"] = normalise_data(clarity_mapping[element["clarity"]], 1.0, 8.0)
        element["depth"] = normalise_data(float(element["depth"]), 43, 79)
        element["table"] = normalise_data(float(element["table"]), 42, 95)
        element["x"] = normalise_data(float(element["x"]), 0, 10.74)
        element["y"] = normalise_data(float(element["y"]), 0, 58.9)
        element["z"] = normalise_data(float(element["z"]), 0, 31.8)
        element["price"] = normalise_data(float(element["price"]), 326, 18823)
        training_inputs.append([element.get(input_key) for input_key in input_keys])
        training_targets.append(float(element["price"]))

    training_inputs = np.array(training_inputs)
    training_targets = np.array(training_targets)

    return training_inputs, training_targets, 326, 18823  # return min and max prices

training_data = parse_csv("M33174_CWK_Data_set.csv")
x = training_data[0]
y = training_data[1]

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

# Initialize weights and biases
np.random.seed(42)
w1 = np.random.randn(x.shape[1], 32)
b1 = np.zeros((1, 32))
w2 = np.random.randn(32, 32)
b2 = np.zeros((1, 32))
w3 = np.random.randn(32, 1)   
b3 = np.zeros((1, 1))

learning_rate = 0.001
epochs = 1000
loss_list = []
for epoch in range(epochs):
    y_pred, z1, a1, z2, a2 = forward(x, w1, b1, w2, b2, w3, b3)
    
    loss = np.mean((y_pred - y.reshape(-1, 1))**2)
    
    grad_loss = 2 * (y_pred - y.reshape(-1, 1)) / len(x)
    
    grad_w3 = np.dot(a2.T, grad_loss)
    grad_b3 = np.sum(grad_loss, axis=0, keepdims=True)
    grad_a2 = np.dot(grad_loss, w3.T)
    
    grad_z2 = grad_a2 * (z2 > 0)
    grad_w2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
    grad_a1 = np.dot(grad_z2, w2.T)
    
    grad_z1 = grad_a1 * (z1 > 0)
    grad_w1 = np.dot(x.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
    
    w3 -= learning_rate * grad_w3
    b3 -= learning_rate * grad_b3
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    
    loss_list.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', label="Loss")
plt.title("Mean Absolute Error Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSQE")
plt.grid(True)
plt.legend()
plt.savefig("part_d_error_plot.png")
plt.show()

# Test the model
test_data = parse_csv("testdata.csv")
x_test = test_data[0]
y_test = test_data[1]

y_pred, _, _, _, _ = forward(x_test, w1, b1, w2, b2, w3, b3)

# Denormalize
y_pred_denorm = denormalise_data(y_pred, 326, 18823)
y_test_denorm = denormalise_data(y_test, 326, 18823)

# Print error
mse = np.mean((y_test_denorm - y_pred_denorm)**2)
print(f"Mean Squared Error: {mse}")

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(y_test_denorm)), y_test_denorm, color='blue', alpha=0.5, label='True Prices')
plt.plot(np.arange(len(y_pred_denorm)), y_pred_denorm, color='red', alpha=0.5, label='Predicted Prices')
plt.legend()
plt.title('Predicted vs Actual Prices')
plt.xlabel('Sample Index')
plt.ylabel('Diamond Price (denormalized)')
plt.savefig('part_d_pred_vs_true.png')
plt.show()

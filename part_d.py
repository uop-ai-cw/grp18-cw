import numpy as np
import csv
import matplotlib.pyplot as plt
import random

class Full_NN:
    def __init__(self, X, HL, Y):
        self.X = X
        self.HL = HL
        self.Y = Y
        L = [X] + HL + [Y]

        # Weight initialization using Xavier Initialization
        self.W = [np.random.randn(L[i], L[i + 1]) * np.sqrt(2 / (L[i] + L[i + 1])) for i in range(len(L) - 1)]
        self.B = [np.zeros((1, L[i + 1])) for i in range(len(L) - 1)]
        self.out = [np.zeros((1, layer)) for layer in L]
        self.Der = [np.zeros_like(w) for w in self.W] 

    def FF(self, x):
        self.out[0] = x.reshape(1, -1)
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            Xnext = np.dot(self.out[i], w) + b
            if i < len(self.W) - 1:
                self.out[i + 1] = self.ReLU(Xnext)  # ReLU for hidden layers
            else:
                self.out[i + 1] = self.sigmoid(Xnext)  # Sigmoid for the output layer
        return self.out[-1]

    def BP(self, Er, lr):
        for i in reversed(range(len(self.Der))):
            out = self.out[i + 1]
            if i == len(self.Der) - 1:
                delta = Er * self.sigmoid_Der(out) # Outer layer
            else:
                delta = Er * self.ReLU_Der(out)  # Hidden layers

            self.Der[i] = np.dot(self.out[i].T, delta)
            self.B[i] = delta.sum(axis=0, keepdims=True)
            
            # Gradient descent
            self.W[i] += self.Der[i] * lr
            self.B[i] += self.B[i] * lr
            
            Er = np.dot(delta, self.W[i].T)

    def train_nn(self, x, target, epochs, lr):
        error_per_epoch = []
        for epoch in range(epochs):
            S_errors = 0
            for j, input in enumerate(x):
                t = target[j].reshape(1, -1)
                output = self.FF(input)
                error = t - output
                S_errors += self.msqe(t, output)
                self.BP(error,lr)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Error: {S_errors / len(x)}")
            error_per_epoch.append(S_errors/len(x))
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(error_per_epoch) + 1), error_per_epoch, marker='o', label="MSQE")
        plt.title("Average MSQE Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSQE")
        plt.grid(True)
        plt.legend()
        plt.savefig("part_d_error_plot.png")
        plt.show()

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_Der(self, x):
        return (x > 0) * 1

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_Der(self, x):
        return x * (1 - x)

    def msqe(self, t, output):
        return np.mean((t - output) ** 2)

def normalise_data(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)

def denormalise_data(data, min_value, max_value):
    return data * (max_value - min_value) + min_value

def parse_csv(csv_path):
    training_inputs = []
    training_targets = []
    test_data = []
    test_target = []

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
    
    # Get test data from dataset
    while len(test_data) < 100:
        random_int = random.randint(0, 53940)
        test_data.append(training_inputs[random_int])
        test_target.append(training_targets[random_int])
        training_inputs.pop(random_int)
        training_targets.pop(random_int)
    
    training_inputs = np.array(training_inputs)
    training_targets = np.array(training_targets)
    test_data = np.array(test_data)
    test_target = np.array(test_target)

    return training_inputs, training_targets, 326, 18823, test_data, test_target

training_inputs, targets, min_price, max_price, test_data, test_target = parse_csv("M33174_CWK_Data_set.csv")

test_data = np.array(test_data)
test_target = np.array(test_target)

nn = Full_NN(9, [64, 32, 16], 1)
nn.train_nn(training_inputs, targets, 100, 0.005)

pred_output = []
real_value = []
error = []

for i in range(len(test_data)):
    output = nn.FF(test_data[i])
    output_true = denormalise_data(output, min_price, max_price)
    output_true = output_true.flatten()[0]
    target_true = denormalise_data(test_target[i], min_price, max_price)
    
    pred_output.append(output_true)
    real_value.append(target_true)
    
    if i % 10 == 0:
        error.append(target_true - output_true)

average_error = np.mean(error)
print(f"Average Error: {average_error}")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(real_value)), real_value, color='blue', alpha=0.5, label='True Prices')
plt.plot(np.arange(len(pred_output)), pred_output, color='red', alpha=0.5, label='Predicted Prices')
plt.legend()
plt.title('Predicted vs Actual Prices')
plt.xlabel('Diamond Samples')
plt.ylabel('Diamond Price (denormalized)')
plt.savefig('part_d_pred_vs_true.png')
plt.show()
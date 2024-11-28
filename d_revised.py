import numpy as np
import csv


class Full_NN(object):
    def __init__(self, X=9, HL=[9, 9, 9, 9, 9], Y=1):
        self.X = X
        self.HL = HL
        self.Y = Y

        L = [X] + HL + [Y]
        L = [X] + HL + [Y]  # total number of layers. This creates a representation of the
        # the network in the format we need it. i.e array of the format [how many inputs, how mnay hidden layers. how many outputs]
        W = []  # initialize a weight array

        for i in range(len(L) - 1):  # we want to be able go to the next layer up so we set one minus
            w = np.random.rand(L[i], L[i + 1])  # fill them up with random values, that is why we need the numpy library
            W.append(w)  # add the new values to the array.
            self.W = W  # link the class variable to the current variable

        # initialize a derivative array. This are needed to calculate the
        # back propagation. they are the derivatives of the activation function
        Der = []
        for i in range(len(L) - 1):
            d = np.zeros((L[i], L[i + 1]))
            Der.append(d)
        self.Der = Der

        out = []  # Output array
        for i in range(len(L)):  # #We don't need to go +1. The outputs are straight forward.
            o = np.zeros(
                L[i])  # we don't need random values, just to have them ready to be used. we fill up with zeros.
            out.append(o)
        self.out = out

    def FF(self, x):
        out = x
        self.out[0] = x
        for i, w in enumerate(self.W):
            out = self.relu(np.dot(out, w))
            self.out[i + 1] = out
        return out

    def BP(self, Er):
        for i in reversed(range(len(self.Der))):
            out = self.out[i + 1]
            D = Er * self.relu_Der(out)
            D_fixed = D.reshape(D.shape[0], -1).T
            this_out = self.out[i].reshape(self.out[i].shape[0], -1)
            self.Der[i] = np.dot(this_out, D_fixed)
            Er = np.dot(D, self.W[i].T)

    def train_nn(self, x, target, epochs, lr):
        for i in range(epochs):
            S_errors = 0
            for j, input in enumerate(x):
                t = target[j]
                output = self.FF(input)
                e = t - output
                self.BP(e)
                self.GD(lr)
                S_errors += self.msqe(t, output)

    def GD(self, lr=0.05):  # Gradient descent
        for i in range(len(self.W)):  # go through the weights
            W = self.W[i]
            Der = self.Der[i]
            W += Der * lr  # update the weights by applying the learning rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_Der(self, x):
        return (x > 0) * 1

    def msqe(self, t, output):
        return np.average((t - output) ** 2)


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


if __name__ == "__main__":
    training_data = parse_csv("M33174_CWK_Data_set.csv")
    training_inputs, training_targets, price_min, price_max = training_data

    nn = Full_NN(9, [9, 9, 9, 9, 9], 1)
    nn.train_nn(training_inputs, training_targets, 10, 0.1)

    input = [0.23, 3.0, 3.0, 5.0, 60.0, 57.0, 4.0, 4.03, 2.41]
    target = 402.0

    NN_output = nn.FF(input)
    NN_output = denormalise_data(NN_output, price_min, price_max)

    print("=============== Testing the Network Screen Output===============")
    print("Test input is ", input)
    print()
    print("Target output is ", target)
    print()
    print("Neural Network actual output is ", NN_output, "there is an error (not MSQE) of ", target - NN_output,
          "Actual = ", target)
    print("=================================================================")

    ##### from the origianl code you sent me, i changed some variable names as it was hard for me to rememeber and removed some comments cos it was had to read aroumnd code

    # changed activation function from sigmoid to relu peeds up convergence during training because it avoids the vanishing gradient problem that is common with sigmoid
    # and relu outputs are simpler

    # denormalised data for "prediction"

    ##explicit data return in parse.cvs

import csv
import numpy as np
#because a lot of data will be needed, we will use a class approach.

class Full_NN(object):
    #A Multi Layer Neural Network class. We use this as for the way we need to handle the
    #variables is better suited.
    def __init__(self, X, HL, Y=2): #a constructor for some default values.
        self.X=X #inputs
        self.HL=HL #hidden layers
        self.Y=Y #outputs

        #we are setting up some class variables for our inputs.
        L=[X]+HL+[Y] #total number of layers. This creates a representation of the
        #the network in the format we need it. i.e array of the format [how many inputs, how mnay hidden layers. how many outputs]
        W=[] #initialize a weight array


        for i in range(len(L)-1): #we want to be able go to the next layer up so we set one minus
            w=np.random.rand(L[i], L[i+1]) #fill them up with random values, that is why we need the numpy library
            W.append(w) #add the new values to the array.
        self.W=W #link the class variable to the current variable
        
        #initialize a derivative array. This are needed to calculate the 
        # back propagation. they are the derivatives of the activation function
        Der=[] 
        for i in range(len(L) - 1):
            #same reason as above for every line
            #we don't need random values, just to have them ready to be used. we fill up with zeros
            d = np.zeros((L[i], L[i+1])) 
            Der.append(d)
        self.Der = Der

        #we will be passing these here as that way the class variable will keep them for us until we need them.
        
        out = [] # Output array
        for i in range(len(L)): # #We don't need to go +1. The outputs are straight forward.
            o = np.zeros(L[i]) #we don't need random values, just to have them ready to be used. we fill up with zeros.
            out.append(o)
        self.out = out

    def FF(self, x): # method will run the network forward
        out = x # input layer output is just the input
        
        self.out[0] = x # Linking the outputs to the class variable for back propagation. Begin with input layer
        
        for i, w in enumerate(self.W): # go through the network layer via the weights variable
            Xnext = np.dot(out,w) # product between weights and output for the next output
            out = self.sigmoid(Xnext) # use the activation function
            self.out[i+1] = out # pass result to the class variable to preserve for later. Back propagation
        
        return out
    
    def BP(self, Er): # Back propagation method. using the output error *Er* to go backwards through the layers and calculate...
        # the errors needed to update the weights
        # This will return the final error of the input
        
        for i in reversed (range(len(self.Der))): # Go backwards *reversed* through the network
            # In this loop we are going backwards through the layers based on the following equations

            out = self.out[i+1] # Get the layer output for the previous layer (we go reverse)
            
            D = Er * self.sigmoid_Der(out) # Applying the derivative of the activation function to get delta.delta
            # This is essentially - dE/DWi =(y - y[i+1]) S'(x[i+1])xi
            
            D_fixed = D.reshape(D.shape[0], -1).T # Turn Delta into an array of appropriate size
            
            this_out = self.out[i] # current layer output
            
            this_out = this_out.reshape(this_out.shape[0], -1) # reshape as before to get column array suitable for the
            # multiplication we need
            
            self.Der[i] = np.dot(this_out, D_fixed) # Matrix multiplication and pass result to class variable
            
            Er = np.dot(D, self.W[i].T) # This back propagates the next error we need for the next iteration
            # this error term is part of the dE/DWi equation for the next layer down in the back propagation
            # and we pass it on after calculating it in this iteration
            
    def train_nn(self, x, target, epochs, lr): # training the network, x = array, target = array, epochs = int, lr = int
        for i in range(epochs): # training loop for as many epochs as we need
            S_errors = 0 # variable to carry the error we need to report to the user
            
            for j, input in enumerate(x): #iterate through the training data and inputs
                t = target[j]
                output = self.FF(input) # use the network calculations for forward calculations
                
                e = t - output # obtain the overall Network output error
                self.BP(e) # use error to do the back propagation
                self.GD(lr) #Do gradient descent
                
                S_errors += self.msqe(t,output) # updates the overall error to show the user
                                
            print(f"Epoch {i + 1}, Loss: {S_errors / len(x)}")
    
    def GD(self, lr=0.05): #Gradient descent
        for i in range(len(self.W)): # go through the weights
            W = self.W[i]
            Der = self.Der[i]
            W += Der * lr # update the weights by applying the learning rate
            
    def sigmoid(self, x): # sigmoid activation function
        y = 1.0/(1 + np.exp(-x))
        return y
    
    def sigmoid_Der(self, x): # sigmoid function derivative
        sig_der = x * (1.0 - x)
        return sig_der
    
    def msqe(self, t, output): # mean square error, gonna try MAE
        # return np.average((t - output) ** 2)
        return np.mean(np.abs(t - output))

def normalise_data(data, feature_min, feature_max):
    return (data - feature_min) / (feature_max - feature_min)

def parse_csv(csv_path):
    training_inputs = []
    training_targets = []

    with open(csv_path, "r") as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]

    # Data processing, need to normalise data 
    input_keys = ["carat","cut","color","clarity","depth","table","x","y","z"]
    cut_mapping = {"Fair": 1.0, "Good": 2.0, "Very Good": 3.0, "Premium": 4.0, "Ideal": 5.0}
    color_mapping = {"D": 7.0, "E": 6.0, "F": 5.0, "G": 4.0, "H": 3.0, "I": 2.0, "J": 1.0}
    clarity_mapping = {
        "I1": 1.0,
        "SI2": 2.0,
        "SI1": 3.0,
        "VS2": 4.0,
        "VS1": 5.0,
        "VVS2": 6.0,
        "VVS1": 7.0,
        "IF": 8.0
    }

    for element in data:
        element["carat"] = normalise_data(float(element["carat"]), 0.2, 5.01)
        element["cut"] = normalise_data(cut_mapping[element["cut"]], 1.0, 5.0)
        element["color"] = normalise_data(color_mapping[element["color"]], 1.0, 5.0)
        element["clarity"] = normalise_data(clarity_mapping[element["clarity"]], 1.0, 8.0)
        element["depth"] = normalise_data(float(element["depth"]), 43.0, 79.0)
        element["table"] = normalise_data(float(element["table"]),42.0, 95.0)
        element["x"] = normalise_data(float(element["x"]), 0, 10.74)
        element["y"] = normalise_data(float(element["y"]), 0, 58.9)
        element["z"] = normalise_data(float(element["z"]), 0, 31.8)
        element["price"] = normalise_data(float(element["price"]), 326.0, 18823.0)
        training_inputs.append([element.get(input_key) for input_key in input_keys])
        training_targets.append(element.get("price"))
    
    training_inputs = np.array(training_inputs)
    training_targets = np.array(training_targets) 
    
    return training_inputs, training_targets

if __name__ == "__main__": # Testing class
    training_data = parse_csv("M33174_CWK_Data_set.csv")
    training_inputs, training_targets = training_data[0], training_data[1]
    # print(training_inputs, training_targets)
    
    split_ratio = 0.001
    train_size = int(training_inputs.shape[0] * split_ratio)
    input_test, input_train = training_inputs[:train_size], training_inputs[train_size:]
    target_test, target_train = training_targets[:train_size], training_targets[train_size:]

    nn = Full_NN(9, [128,64,32,16], 1) # Creates a NN with 9 inputs, 2 hidden layers and 1 output
    nn.train_nn(input_train, target_train, 100, 0.01) # Train network with 0.1 learning rate for 10 epochs
    
    """ input = np.array([normalise_data(0.23, 0.2, 5.01),
             normalise_data(5.0, 1.0, 5.0),
             normalise_data(6.0, 1.0, 5.0),
             normalise_data(2.0, 1.0, 8.0),
             normalise_data(61.5, 43.0, 79.0),
             normalise_data(55.0,42.0, 95.0),
             normalise_data(3.95, 0, 10.74),
             normalise_data(3.98, 0, 58.9),
             normalise_data(2.43, 0, 31.8)
             ])  """# After training this tests the train network
    
    test_inputs = np.array(input_test)
    test_targets = np.array(target_test)
    
    target = np.array(normalise_data(326.0, 326.0, 18823.0)) # Target value
    target_real = (target * (18823 - 326)) + 326
    
    """ test_outputs = []
    for i in range(len(input_test) - 1):
        formatted = np.array(input_test[i])
        output = nn.FF(formatted).item()
        print(formatted,output)
    print(test_outputs) """
    
    NN_output = nn.FF(input_test[4])
    print(NN_output, target_test[4])
    
    """ print("=============== Testing the Network Screen Output===============")
    for i in range(len(input_train)-1):
        print(f"Input: {input_test[i]}, Target: {target_test[i]}")
    print("=================================================================")
    print(len(input_train), len(input_test)) """
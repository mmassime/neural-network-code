import numpy as np
import scipy

class ANN():
    def __init__(self, n_input,n_hidden_layers,n_hidden_neurons, n_output, learning_rate):
        self.n_input = n_input
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.weights = []
        #self.w_input_hidden = np.random.rand(n_hidden_neurons, n_input) - 0.5
        self.weights.append(np.random.rand(n_hidden_neurons, n_input) - 0.5)
        for n in range(self.n_hidden_layers-1):
            self.weights.append(np.random.rand(n_hidden_neurons, n_hidden_neurons) - 0.5)
        self.weights.append(np.random.rand(n_output, n_hidden_neurons) - 0.5)
        #self.w_hidden_output = np.random.rand(n_output, n_hidden_neurons) - 0.5
        self.activation_function = lambda x: scipy.special.expit(x)
    def print(self):
        for i in range(len(self.weights)):
            print("layer number :", i)
            print(self.weights[i].shape)
            print(self.weights[i][:10])
    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        outputs = []
        outputs.append(inputs)
        #outputs.append(self.activation_function(np.dot(self.weights[0], inputs)))
        for layer in self.weights:
            outputs.append(self.activation_function(np.dot(layer, outputs[-1])))
        output_errors = targets - outputs[-1]
        current_errors = np.dot(self.weights[-1].T, output_errors)
        self.weights[-1] += self.learning_rate * np.dot((output_errors * outputs[-1] * (1.0 - outputs[-1])),
        np.transpose(outputs[-2]))
        output_errors = current_errors
        for i in range(len(self.weights)-2,-1,-1):
            current_errors = np.dot(self.weights[i].T, output_errors)
            self.weights[i] += self.learning_rate * np.dot((output_errors * outputs[i+1] * (1.0 - outputs[i+1])),
            np.transpose(outputs[i]))
            output_errors = current_errors

        #self.weigths[0] += self.learning_rate * np.dot((hidden_errors * outputs[1] * (1.0 - outputs[1])), np.transpose(inputs))
    def Relu(self, inputs):
        return np.maximum(0,inputs)
    def predict(self, I):
        inputs = np.array(I, ndmin=2).T
        output = self.activation_function(np.dot(self.weights[0], inputs))
        for layer in self.weights[1:]:
            output = self.activation_function(np.dot(layer, output))
        #output_hidden = self.activation_function(np.dot(self.w_input_hidden, I))
        #output_final = self.activation_function(np.dot(self.w_hidden_output, output_hidden))
        return output
    def accuracy(self, pred, Y):
        pred = np.array(pred)
        Y = np.array(Y)
        return np.sum(pred == Y) / Y.size
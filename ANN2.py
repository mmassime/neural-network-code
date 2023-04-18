import numpy as np
import scipy

class ANN():
    def __init__(self, n_input, n_hidden_neurons, n_output, learning_rate):
        self.n_input = n_input
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.w_input_hidden = np.random.rand(n_hidden_neurons, n_input) - 0.5
        self.w_hidden_output = np.random.rand(n_output, n_hidden_neurons) - 0.5
        self.activation_function = lambda x: scipy.special.expit(x)
    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        output_hidden = self.activation_function(np.dot(self.w_input_hidden, inputs))
        final_outputs = self.activation_function(np.dot(self.w_hidden_output, output_hidden))
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_hidden_output.T, output_errors)
        self.w_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        np.transpose(output_hidden))
        self.w_input_hidden += self.learning_rate * np.dot((hidden_errors * output_hidden * (1.0 - output_hidden)), np.transpose(inputs))
    def Relu(self, inputs):
        return np.maximum(0,inputs)
    def predict(self, I):
        inputs = np.array(I, ndmin=2).T
        output_hidden = self.activation_function(np.dot(self.w_input_hidden, I))
        output_final = self.activation_function(np.dot(self.w_hidden_output, output_hidden))
        return output_final
    def accuracy(self, pred, Y):
        pred = np.array(pred)
        Y = np.array(Y)
        return np.sum(pred == Y) / Y.size
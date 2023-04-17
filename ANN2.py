import numpy as np

class ANN():
    def __init__(self, n_input, n_hidden_neurons, n_output, learning_rate):
        self.n_input = n_input
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.w_input_hidden = np.random.rand(n_hidden_neurons, n_input) - 0.5
        self.w_hidden_output = np.random.rand(n_output, n_hidden_neurons) - 0.5
    def train():
        pass
    def Relu(self, inputs):
        return np.maximum(0,inputs)
    def predict(self, I):
        output_hidden = self.Relu(np.dot(self.w_hidden_output, I))
        output_final = self.Relu(np.dor(self.w_hidden_output, output_hidden))

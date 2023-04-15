import numpy as np

class Neuron:
    def __init__(self, n_weights) -> None:
        self.bias = np.random.rand()
        self.n_weights = n_weights
        self.weights = np.random.rand(n_weights)
    def __call__(self, inputs: np.array) -> float:
        product = np.dot(self.weights, inputs)
        sum = np.sum(product) + self.bias
        return sum
    
class ANN:
    def __init__(self, n_input, n_hidden_layers, n_hidden_neurons, n_outputs):
        self.n_input = n_input
        self.inputs = np.zeros(n_input)
        self.n_hidden_layers = n_hidden_layers
        previous_n = n_input
        self.layers = []
        for i in range(n_hidden_layers):
            layer = [Neuron(previous_n) for i in range(n_hidden_neurons)]
            previous_n = n_hidden_neurons
            self.layers.append(layer)
        self.n_hidden_neurons = n_hidden_neurons
        self.n_outputs = n_outputs
        self.output_layer = [Neuron(previous_n) for i in range(n_outputs)]
    
    def Relu(self, inputs):
        return np.maximum(0,inputs)

    def SoftMax(self, inputs):
        return np.exp(inputs) / np.sum(np.exp(inputs))

    def forward(self, inputs):
        prev_layer = inputs
        output = []
        for layer in self.layers:
            output = [n(prev_layer) for n in layer]
            prev_layer = self.Relu(output)
        final_output = [n(prev_layer) for n in self.output_layer]    
        final_output = self.SoftMax(final_output)
        return final_output

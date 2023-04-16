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
    def __init__(self, n_input, n_hidden_layers, n_hidden_neurons, n_outputs, learning_rate):
        self.learning_rate = learning_rate
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
    
    def DRelu(self, input):
        return input > 0
    
    def SoftMax(self, inputs):
        return np.exp(inputs) / np.sum(np.exp(inputs))

    def forward(self, inputs, expected_output):
        prev_layer = inputs
        outputs = []
        output = []
        for layer in self.layers:
            output = [n(prev_layer) for n in layer]
            prev_layer = self.Relu(output)
            outputs.append(prev_layer)
        final_output = [n(prev_layer) for n in self.output_layer]    
        final_output = self.SoftMax(final_output)
        max_res = np.max(final_output)
        res = np.where(final_output == max_res, 1, 0)
        outputs.append(res)
        loss = res - expected_output
        loss = np.sum(np.power(loss, 2))
        d = self.backProp(outputs, expected_output)
        return outputs, d
    
    def backProp(self, outputs, expected_outptus):
        d_layer = []
        for i,o in enumerate(outputs[-1]):
            derivative = 2 * (o-expected_outptus[i])
            d_layer.append(derivative)
        d_layers = []
        d_layers.insert(0,d_layer)
        next_layer = self.output_layer
        d_next_layer = d_layer
        for i in range(self.n_hidden_layers-1, -1,-1):
            d_layer = []
            for j in range(len(self.layers[i])):
                derivative = 0
                for k in range(len(next_layer)):
                    derivative += d_next_layer[k]*next_layer[k].weights[j]
                d_layer.append(derivative * self.DRelu(outputs[i][j]))
            next_layer = self.layers[i]
            d_next_layer = d_layer
            d_layers.insert(0, d_layer)
        return d_layers
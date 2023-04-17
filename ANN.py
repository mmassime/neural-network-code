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
        self.inputs = inputs
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
        self.backProp(outputs, expected_output)
    
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
        prev_input = self.inputs
        d_layer_weights = []
        for i in range(len(self.layers)):
            d_weights = []
            for j in range(len(self.layers[i])):
                d_weight = []
                for k in range(len(prev_input)):
                    d_weight.append(prev_input[k]*d_layers[i][j])
                d_weights.append(d_weight)
            d_layer_weights.append(d_weights)
            prev_input = d_layers[i]
        d_weights = []
        for j in range(len(self.output_layer)):
            d_weight = []
            for k in range(len(prev_input)):
                d_weight.append(prev_input[k]*d_layers[-1][j])
            d_weights.append(d_weight)
        d_layer_weights.append(d_weights)

        self.updateParams(d_layers, d_layer_weights)
    
    def updateParams(self, d_layers, d_weights):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                node = self.layers[i][j]
                node.bias -= self.learning_rate*d_layers[i][j]
                for k in range(len(node.weights)):
                    node.weights[k] -= self.learning_rate*d_weights[i][j][k]
        for j in range(len(self.output_layer)):
            node = self.output_layer[j]
            node.bias -= self.learning_rate*d_layers[-1][j]
            for k in range(len(node.weights)):
                node.weights[k] -= self.learning_rate*d_weights[-1][j][k]
    def toHotOne(self, y):
        res = [0]*self.n_outputs
        res[y] = 1
        return res
    def toClass(self, y):
        return np.where(y==1)
    
    def accuracy(self, pred, Y):
        pred = np.array(pred)
        Y = np.array(Y)
        return np.sum(pred == Y) / Y.size
    
    def training(self, X, Y, iterations):
        for idx, layer in enumerate(self.layers):
            print("layer n :" + str(idx))
            for i, n in enumerate(layer):
                print("for neuron n: " + str(i))
                print("weights = " + str(n.weights[:10]))
                print("bias = " + str(n.bias))
        print("output layer")
        for i, n in enumerate(self.output_layer):
            print("weights = " + str(n.weights))
            print("bias = " + str(n.bias))
        for i in range(iterations):
            print(i)
            for x,y in zip(X,Y):
                res = self.toHotOne(y)
                self.forward(x, res)
               
            if i % 10 == 0:
                predictions = self.predict(X)
                print("at iteration :", i)
                print("accuracy : ", self.accuracy(predictions, Y))
                for idx, layer in enumerate(self.layers):
                    print("layer n :" + str(idx))
                    for i, n in enumerate(layer):
                        print("for neuron n: " + str(i))
                        print("weights = " + str(n.weights[:10]))
                        print("bias = " + str(n.bias))
                print("output layer")
                for i, n in enumerate(self.output_layer):
                    print("weights = " + str(n.weights))
                    print("bias = " + str(n.bias))
    def predict(self, X):
        predictions = []
        for x in X:
            prev_layer = x
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
            predictions.append(res[0])
        return predictions
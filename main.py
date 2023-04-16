from ANN import ANN
import numpy as np
neural_network = ANN(2, 2, 2, 2, 0.1)

for idx, layer in enumerate(neural_network.layers):
    print("layer n :" + str(idx))
    for i, n in enumerate(layer):
        print("for neuron n: " + str(i))
        print("weights = " + str(n.weights))
        print("bias = " + str(n.bias))
print("output layer")
for i, n in enumerate(neural_network.output_layer):
    print("weights = " + str(n.weights))
    print("bias = " + str(n.bias))
inputs = np.array([1,1])
outputs, d= neural_network.forward(inputs, [1,0])
print(outputs, d)
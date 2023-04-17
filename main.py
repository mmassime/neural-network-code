from ANN2 import ANN
import numpy as np
#from keras.datasets import mnist
"""
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(60000, 784)
test_X = test_X.reshape(10000,784)
neural_network = ANN(784, 1, 10,10,0.1)

neural_network.training(train_X[:1000], train_y[:1000],1000)
"""
neural_network = ANN(2,2,2, 0.1)
print(neural_network.predict([1,1]))
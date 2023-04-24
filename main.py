from ANN import ANN
import numpy as np
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(60000, 784)
test_X = test_X.reshape(10000,784)
neural_network = ANN(784,1,150,10,0.1)
epochs = 3
for e in range(epochs):
    for x,y in zip(train_X, train_y):
        # preprocessing of the data
        inputs = (np.asfarray(x) / 255.0 * 0.99) + 0.01
        targets = np.zeros(10) + 0.01
        targets[y] = 0.99
        neural_network.train(inputs, targets)
    predictions = []
    #after every epoch print the accuracy on test data
    for x in test_X:
        pred = neural_network.predict(x)
        maxPred = np.max(pred)
        pred = np.where(pred==maxPred)
        predictions.append(pred[0][0])
    print("the accuracy is : ", neural_network.accuracy(predictions,test_y))

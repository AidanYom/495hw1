from learningUtils import *
from random import shuffle

learningRate = .5

numIterations = 100

inputDim = 2
outputDim = 1

hiddenLayerDim = 4
#two layers with a sufficient number of hidden nodes
#if the input is n, and output is 2^n, it can learn anything
#this is only for binary inputs

scale = 100

X = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] * scale
shuffle(X)

split = int((len(X) * 3) / 4)

X_train = [x[0:2] for x in X[0:split]]
X_test = [x[0:2] for x in X[split:]]
y_train = [x[2:] for x in X[0:split]]
y_test = [x[2:] for x in X[split:]]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
w1, b1, w2, b2 = initialize(inputDim, outputDim, hiddenLayerDim)

for iterNum in range(1, numIterations + 1):
    Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_train)

    if iterNum  % 5 == 0:
        printLoss(y_train, A2, iterNum)

    gradWeights, gradBiases, gradWeights1, gradBiases1 = backPropagation(Z1, A1, Z2, A2, w1, w2, X_train, y_train)

    w2, b2 = updateParams(w2, b2, gradWeights, gradBiases, learningRate)
    w1, b1 = updateParams(w1, b1, gradWeights1, gradBiases1, learningRate)


Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_test)

printAccuracy(A2, y_test)
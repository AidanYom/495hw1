from learningUtils import *
from random import shuffle

learningRate = 1

numIterations = 20

inputDim = 2
outputDim = 1

hiddenLayerDim = 10

scale = 5

X = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] * scale
shuffle(X)

split = int((len(X) * 4) / 5)

X_train = [x[0:2] for x in X[0:split]]
X_test = [x[0:2] for x in X[split:]]
y_train = [x[2:] for x in X[0:split]]
y_test = [x[2:] for x in X[split:]]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
w1, b1, w2, b2 = initialize(inputDim, outputDim, hiddenLayerDim)

for iterNum in range(numIterations):
    Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_train)

    aggLoss = 0
    for i in range(len(A2)):
        aggLoss += loss(y_train[i], A2[i])

    print("iteration:", iterNum, "loss:", aggLoss)

    aggGradLoss = [0] * len(A2[0])

    for i in range(len(A2)):
        aggGradLoss = vAdd(aggGradLoss, vMultiplyConstant((vSub(A2[i], y_train[i])), 2 / len(A2)))
    
    aggLoss = loss(A2[i], y_train[i])
    
    aggGradSigmoid = [0] * len(Z2[0])

    for i in range(len(Z2)):
        aggGradSigmoid = vAdd(aggGradSigmoid, vMultiplyConstant(sigmoid_layer_derivative(Z2[i]), 1 / len(Z2)))

    gradBiases = vMult(aggGradLoss, aggGradSigmoid)

    gradWeights = mMult(gradBiases, vMultiplyConstant(A1[0], 1 / len(A1)))

    for i in range(1, len(A1)):
        gradWeights = mAdd(gradWeights, mMult(gradBiases, vMultiplyConstant(A1[i], 1 / len(A1))))

    aggGradSigmoid1 = [0] * len(Z1[0])

    for i in range(len(Z1)):
        aggGradSigmoid1 = vAdd(aggGradSigmoid1, vMultiplyConstant(sigmoid_layer_derivative(Z1[i]), 1 / len(Z1)))
    
    gradBiases1 = vMult(mMultiplyVector(transpose(w2), gradBiases), aggGradSigmoid1)
    
    gradWeights1 = mMult(gradBiases1, vMultiplyConstant(X_train[0], 1 / len(X_train)))

    for i in range(1, len(X_train)):
        gradWeights1 = mAdd(gradWeights1, mMult(gradBiases1, vMultiplyConstant(X_train[i], 1 / len(X_train))))

    w2, b2 = updateParams(w2, b2, gradWeights, gradBiases, learningRate)
    w1, b1 = updateParams(w1, b1, gradWeights1, gradBiases1, learningRate)


Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_test)

predictions = [int(a[0] > .5) for a in A2]

num_correct = 0

for pred, actual in zip(predictions, y_test):
    if(pred == actual[0]):
        num_correct += 1

accuracy = num_correct / len(predictions) * 100

print("accuracy:", accuracy, "%")
from learningUtils import *
from random import shuffle

learningRate = 0.1

numIterations = 1000

inputDim = 2
outputDim = 1

hiddenLayerDim = 4

scale = 1

X = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] * scale
shuffle(X)

split = int((len(X) * 3) / 4)

X_train = [x[0:2] for x in X[0:split]]
X_test = [x[0:2] for x in X[split:]]
y_train = [x[2:] for x in X[0:split]]
y_test = [x[2:] for x in X[split:]]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
w1, b1, w2, b2 = initialize(inputDim, outputDim, hiddenLayerDim)

# print("w1, b1, w2, b2")
# print(w1)
# print(b1)
# print(w2)
# print(b2)
# print()


for i in range(numIterations):
    Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_train)
    # print("z1, a1, z2, a2")
    # print(Z1[0])
    # print(A1[0])
    # print(Z2[0])
    # print(A2[0])

    # print()
    # # print(Z1)
    # print(A2)
    # print(y_train)

    #grad loss is a vector of the same size of the output layer
    #it is the average of the gradient of the loss over each training sample
    aggGradLoss = [0] * len(A2[0])

    for i in range(len(A2)):
        aggGradLoss = vAdd(aggGradLoss, vMultiplyConstant((vSub(A2[i], y_train[i])), 2 / len(A2)))
    
    aggLoss = loss(A2[i], y_train[i])
    # print("aggGrad loss, aggGrad sigmoid")
    # print(aggGradLoss)

    
    aggGradSigmoid = [0] * len(Z2[0])

    # print(A1)
    for i in range(len(Z2)):
        aggGradSigmoid = vAdd(aggGradSigmoid, vMultiplyConstant(sigmoid_layer_derivative(Z2[i]), 1 / len(Z2)))

    # print(aggGradSigmoid)

    # print()
    # print(A1[0])
    # print(w2)
    # print(Z2[0])

    #this vector multiplication should give the amount we shift the biases by
    #though it will still need to be multiplied by the learning rate
    gradBiases = vMult(aggGradLoss, aggGradSigmoid)

    # print()
    # print("grad biases, grad weights")
    # print(gradBiases)

    gradWeights = mMult(gradBiases, vMultiplyConstant(A1[i], 1 / len(A1)))

    for i in range(1, len(A1)):
        gradWeights = mAdd(gradWeights, mMult(gradBiases, vMultiplyConstant(A1[i], 1 / len(A1))))

    # print(gradWeights)



    #update parameters
    

    aggGradSigmoid1 = [0] * len(Z1[0])

    # print(A1)
    for i in range(len(Z1)):
        aggGradSigmoid1 = vAdd(aggGradSigmoid1, vMultiplyConstant(sigmoid_layer_derivative(Z1[i]), 1 / len(Z1)))
    # print()
    # print(gradWeights)
    # print(aggGradSigmoid1)
    # gradBiases1 = mMultiplyVector(gradWeights, aggGradSigmoid1)
    # print(gradBiases1)
    # print(b1)
    # aggGradLoss1 = w2
    # print()
    # print(transpose(w2))
    
    gradBiases1 = vMult(mMultiplyVector(transpose(w2), gradBiases), aggGradSigmoid1)
    # print(gradBiases)
    # print()
    # print(b1)
    gradWeights1 = mMult(gradBiases1, vMultiplyConstant(X_train[i], 1 / len(X_train)))

    for i in range(1, len(X_train)):
        gradWeights1 = mAdd(gradWeights1, mMult(gradBiases1, vMultiplyConstant(X_train[i], 1 / len(X_train))))

    # print(w1)
    # print(gradWeights1)

    w2, b2 = updateParams(w2, b2, gradWeights, gradBiases, learningRate)
    w1, b1 = updateParams(w1, b1, gradWeights1, gradBiases1, learningRate)


Z1, A1, Z2, predictions = forward_prop(w1, b1, w2, b2, X_test)
print(predictions)
print(y_test)
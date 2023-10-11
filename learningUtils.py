from math import e
from random import uniform

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
def vAdd(u, v):
    return [u[i]+v[i] for i in range(len(u))]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
def vSub(u, v):
    return [u[i]-v[i] for i in range(len(u))]

def vSubK(u, k):
    return [u[i]-k for i in range(len(u))]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
def vMultiplyConstant(u, k): 
    return [k*u[i] for i in range(len(u))]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
def mMultiplyVector(m, v): 
    return [vDot(m[i], v) for i in range(len(m))]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
def vDot(u, v):
    sum = 0
    for i in range(len(u)): 
        sum += u[i]*v[i]
    return sum

def vMult(u, v):
    return [u[i]*v[i] for i in range(len(u))]

def mAdd(u, v):
    for col in range(len(v)):
        for idx in range(len(v[col])):
            u[col][idx] += v[col][idx]

    return u

def transpose(m):
    transMatrix = []
    for i in range(len(m[0])):
        temp = []
        for j in range(len(m)):
            temp.append(m[j][i])
        transMatrix.append(temp)

    return transMatrix

def mSub(u, v):
    for col in range(len(v)):
        for idx in range(len(v[col])):
            u[col][idx] -= v[col][idx]

    return u

def mMult(u,v):
    prod = []
    for u_term in u:
        temp = []
        for v_term in v:
            temp.append(u_term * v_term)
        prod.append(temp)
    
    return prod

def mMultConstant(u, k):
    prod = []
    for col in u:
        temp = []
        for row in col:
            temp.append(row * k)
        prod.append(temp)

    return prod

def sigmoid(x):
    return 1 / ((e ** (-x)) + 1)

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
def sigmoid_layer(X): 
    return [sigmoid(X[i]) for i in range(len(X))]

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_layer_derivative(X): 
    return [sigmoid_derivative(X[i]) for i in range(len(X))]

# adapted from https://github.com/qobi/ece49595cv/blob/main/two_layer_perceptron.py
def loss(u, v):
    return vDot(vSub(u, v), vSub(u, v))

def fc_layer(X, weights, biases):
    return vAdd(mMultiplyVector(weights, X), biases)


def initialize(inputDim, outputDim, hiddenLayerDim):
    w1 = [[uniform(-1, 1) for i in range(inputDim)]
            for j in range(hiddenLayerDim)]
    b1 = [uniform(-1, 1) for j in range(hiddenLayerDim)]
    w2 = [[uniform(-1, 1) for j in range(hiddenLayerDim)] for j in range(outputDim)]
    b2 = [uniform(-1, 1) for i in range(outputDim)]

    return w1, b1, w2, b2

def mAvg(m):
    newM = [0] * len(m[0])
    for col in m:
        for rowIdx in range(len(col)):
            newM[rowIdx] += col[rowIdx] / len(col)

    return newM

def forward_prop(w1, b1, w2, b2, X):
    Z1, A1, Z2, A2 = [], [], [], []

    for i in range(len(X)):
        Z1.append(fc_layer(X[i], w1, b1))
        A1.append(sigmoid_layer(Z1[i]))
        Z2.append(fc_layer(A1[i], w2, b2))
        A2.append(sigmoid_layer(Z2[i]))

    return Z1, A1, Z2, A2

def updateParams(initialW, initialB, gradW, gradB, learning_rate):
    newW = mSub(initialW, mMultConstant(gradW, learning_rate))
    newB = vSub(initialB, vMultiplyConstant(gradB, learning_rate))

    return newW, newB

def printLoss(y_train, A2, iterNum):
    aggLoss = 0
    for i in range(len(A2)):
        aggLoss += loss(y_train[i], A2[i])

    print("iteration:", iterNum, "loss:", aggLoss)

def backPropagation(Z1, A1, Z2, A2, w1, w2, X, Y):

#calculations for the output layer
    
    #calculate the aggregate of the gradient of the loss function across all inputs of X
    #this is equal to the 1/num_samples * SUMMATION(2 * (Prediction - Output))
    aggGradLoss = [0] * len(A2[0])
    for i in range(len(A2)):
        aggGradLoss = vAdd(aggGradLoss, vMultiplyConstant((vSub(A2[i], Y[i])), 2 / len(A2)))
    
    #calculate the aggregate of the gradient of the sigmoid function across all inputs X
    #this is equal to 1/num_samples * SUMMATION(derivative of the sigmoid evaluated at each point in Z2)
    #Where Z2 is the output of the fully connected linear layer before the activation function
    aggGradSigmoid = [0] * len(Z2[0])
    for i in range(len(Z2)):
        aggGradSigmoid = vAdd(aggGradSigmoid, vMultiplyConstant(sigmoid_layer_derivative(Z2[i]), 1 / len(Z2)))

    #calculate the gradient of the output with respect to the bias
    #from chain rule, this is the product of the aggregate gradient loss, and the aggregate gradient of the sigmoid
    #this loop calculates the aggregate weight across all X
    gradBiases = vMult(aggGradLoss, aggGradSigmoid)

    #calculate the gradient of the output with respect to the weights
    #from chain rule, this is the gradient of the biases, multiplied by the output of the previous layer
    #this loop calculates the aggregate weight across all X
    gradWeights = mMult(gradBiases, vMultiplyConstant(A1[0], 1 / len(A1)))
    for i in range(1, len(A1)):
        gradWeights = mAdd(gradWeights, mMult(gradBiases, vMultiplyConstant(A1[i], 1 / len(A1))))

#calculation for the hidden layer

    #calculate the aggregate of the gradient of the sigmoid layer across all inputs X
    #Z1 is the output of the fully connected linear layer from the hidden layer, before activation function
    #this follows the same process as above, but with the outputs from the hidden layer
    aggGradSigmoid1 = [0] * len(Z1[0])
    for i in range(len(Z1)):
        aggGradSigmoid1 = vAdd(aggGradSigmoid1, vMultiplyConstant(sigmoid_layer_derivative(Z1[i]), 1 / len(Z1)))
    
    #calculate the gradient of the output with respect to the biases of the first hidden layer
    #this is the product of W2 (the weights of the output layer) transposed, and the 
    #the gradient of the output with respect to the bias of the output layer, multiplied by the 
    #previous sigmoid gradient calculated for this stage
    gradBiases1 = vMult(mMultiplyVector(transpose(w2), gradBiases), aggGradSigmoid1)
    
    #this is the gradient of the output with respect to the weights of the first layer
    #this is the aggregate of the product of the gradient of the output with respect to the first
    #biases, and the inputs
    gradWeights1 = mMult(gradBiases1, vMultiplyConstant(X[0], 1 / len(X)))
    for i in range(1, len(X)):
        gradWeights1 = mAdd(gradWeights1, mMult(gradBiases1, vMultiplyConstant(X[i], 1 / len(X))))

    print()
    print(gradWeights)
    print()
    print(gradBiases)
    print()
    print(gradWeights1)
    print()
    print(gradBiases1)
    print()

    return gradWeights, gradBiases, gradWeights1, gradBiases1

def printAccuracy(test_pred, y_test):
    predictions = [int(a[0] > .5) for a in test_pred]

    num_correct = 0

    for pred, actual in zip(predictions, y_test):
        if(pred == actual[0]):
            num_correct += 1

    accuracy = num_correct / len(predictions) * 100

    print("accuracy:", accuracy, "%")
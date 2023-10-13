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

    # print("iteration:", iterNum, "| Average loss:", aggLoss / len(A2))
    return aggLoss / len(A2)

def backPropagation(Z1, A1, Z2, A2, w1, w2, X, Y):

#calculations for the output layer
#stochastic gradient descent, send each input and alter each weight

    #calculate the gradient of the loss function across X
    #this is equal to(2 * (Prediction - Output))
    gradLoss = vMultiplyConstant((vSub(A2, Y)), 2)
    
    #calculate the gradient of the sigmoid function across X
    #this is equal to (derivative of the sigmoid evaluated at each point in Z2)
    #Where Z2 is the output of the fully connected linear layer before the activation function
    gradSigmoid = sigmoid_layer_derivative(Z2)

    #calculate the gradient of the output with respect to the bias
    #from chain rule, this is the product of the aggregate gradient loss, and the aggregate gradient of the sigmoid
    #this loop calculates the aggregate weight across all X
    gradBiases = vMult(gradLoss, gradSigmoid)

    #calculate the gradient of the output with respect to the weights
    #from chain rule, this is the gradient of the biases, multiplied by the output of the previous layer
    gradWeights = mMult(gradBiases, A1)

#calculation for the hidden layer

    #calculate the aggregate of the gradient of the sigmoid layer across all inputs X
    #Z1 is the output of the fully connected linear layer from the hidden layer, before activation function
    #this follows the same process as above, but with the outputs from the hidden layer
    gradSigmoidNext = sigmoid_layer_derivative(Z1)
    
    #calculate the gradient of the output with respect to the biases of the first hidden layer
    #this is the product of W2 (the weights of the output layer) transposed, and the 
    #the gradient of the output with respect to the bias of the output layer, multiplied by the 
    #previous sigmoid gradient calculated for this stage
    gradBiasesNext = vMult(mMultiplyVector(transpose(w2), gradBiases), gradSigmoidNext)
    
    #this is the gradient of the output with respect to the weights of the first layer
    #this is the gradient of the output with respect to the first
    #biases, and the inputs
    gradWeightsNext = mMult(gradBiasesNext, X)

    return gradWeights, gradBiases, gradWeightsNext, gradBiasesNext

def splitInput(input, split, line):
    X_train = [x[0:line] for x in input[0:split]]
    X_test = [x[0:line] for x in input[split:]]
    y_train = [x[line:] for x in input[0:split]]
    y_test = [x[line:] for x in input[split:]]

    return X_train, X_test, y_train, y_test


def printAccuracy(test_pred, y_test):
    predictions = [int(a[0] > .5) for a in test_pred]

    num_correct = 0

    for pred, actual in zip(predictions, y_test):
        if(pred == actual[0]):
            num_correct += 1

    accuracy = num_correct / len(predictions) * 100

    return accuracy

    # print("accuracy:", accuracy, "%")

def trainAndTest(inputDim, outputDim, hiddenLayerDim, X_train, X_test, y_train, y_test, name, learningRate, numIterations):
    # print("Training and testing of", name,"\n")
    # print("The parameters are as follows:")
    # print("Number of hidden layers: 1")
    # print("Hidden layer(s) dimension:", hiddenLayerDim)
    # print("Learning Rate:", learningRate)
    # print("Number of epochs: ", numIterations)
    # print("Samples used: ", len(X_train) + len(X_test))
    # print("Training and Test Split: 80/20")
    # print()


    w1, b1, w2, b2 = initialize(inputDim, outputDim, hiddenLayerDim)

    for iterNum in range(1, numIterations + 1):
        Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_train)

        if ((iterNum  % (numIterations / 2) == 0) or (iterNum == 1)):
            loss = printLoss(y_train, A2, iterNum)

        netGW2 = w2
        netGB2 = b2
        netGW1 = w1
        netGB1 = b1

        for i in range(len(X_train)):
            gradWeights2, gradBiases2, gradWeights1, gradBiases1 = backPropagation(Z1[i], A1[i], Z2[i], A2[i], w1, w2, X_train[i], y_train[i])
            netGW2, netGB2 = updateParams(w2, b2, gradWeights2, gradBiases2, learningRate)
            netGW1, netGB1 = updateParams(w1, b1, gradWeights1, gradBiases1, learningRate)

        w2 = netGW2
        b2 = netGB2
        w1 = netGW1
        b1 = netGB1


    Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X_test)

    acc = printAccuracy(A2, y_test)
    # print("\n")
    return loss, acc

#adapted from chat.openai.com
def generate_two_bit_adder_dataset():
    dataset = []
    for a0 in range(2):
        for b0 in range(2):
            for c0 in range(2):
                for a1 in range(2):
                    for b1 in range(2):
                        # Calculate the outputs
                        s0 = (a0 ^ b0) ^ c0
                        s1 = (a1 ^ b1) ^ (s0 & c0)
                        c2 = (s0 & c0) | (a1 & b1)

                        # Append the inputs and outputs as a list
                        dataset.append([a0, b0, c0, a1, b1, s0, s1, c2])

    return dataset


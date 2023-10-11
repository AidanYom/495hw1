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

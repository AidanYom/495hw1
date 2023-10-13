from learningUtils import *
from random import shuffle

learningRate = .01
numIterations = 1000

XOR_inputDim = 2
XOR_outputDim = 1

XOR_hiddenLayerDim = 4
#two layers with a sufficient number of hidden nodes
#if the input is n, and output is 2^n, it can learn anything
#this is only for binary inputs

scale = 100

XOR = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] * scale
shuffle(XOR)

split = int((len(XOR) * 4) / 5)

XOR_X_train, XOR_X_test, XOR_y_train, XOR_y_test = splitInput(XOR, split, 2)



adder_inputDim = 5
adder_outputDim = 3
adder_hiddenLayerDim = 32

adder = generate_two_bit_adder_dataset() * scale
shuffle(adder)
adder_X_train, adder_X_test, adder_y_train, adder_y_test = splitInput(adder, split, 5)

# for i in range(1, 4):
    # for learningRate in [.01, .2, .5]: 
    #     for numIterations in [10, 100, 1000]:

trainAndTest(XOR_inputDim, XOR_outputDim, XOR_hiddenLayerDim, XOR_X_train, XOR_X_test, XOR_y_train, XOR_y_test, "XOR", learningRate, numIterations)

trainAndTest(adder_inputDim, adder_outputDim, adder_hiddenLayerDim, adder_X_train, adder_X_test, adder_y_train, adder_y_test, "Two Bit Adder", learningRate, numIterations)

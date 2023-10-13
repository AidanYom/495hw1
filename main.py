from learningUtils import *
from random import shuffle
import time

print("Running Aidan Anastario's HW1...\n")
numIterations = 1000

XOR_inputDim = 2
XOR_outputDim = 1

XOR_hiddenLayerDim = 4
XOR_learningRate = .1
#two layers with a sufficient number of hidden nodes
#if the input is n, and output is 2^n, it can learn anything
#this is only for binary inputs

XOR_scale = 100

XOR = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] * XOR_scale
shuffle(XOR)

split = int((len(XOR) * 4) / 5)

XOR_X_train, XOR_X_test, XOR_y_train, XOR_y_test = splitInput(XOR, split, 2)


adder_inputDim = 5
adder_outputDim = 3
adder_hiddenLayerDim = 64

adder_scale = 25
adder_learningRate = .01

adder = generate_two_bit_adder_dataset() * adder_scale
shuffle(adder)

adder_X_train, adder_X_test, adder_y_train, adder_y_test = splitInput(adder, split, 5)

start = time.time()

xorLoss, xorAcc = trainAndTest(XOR_inputDim, XOR_outputDim, XOR_hiddenLayerDim , XOR_X_train, XOR_X_test, XOR_y_train, XOR_y_test, "XOR", XOR_learningRate, numIterations)     

addLoss, addAcc = trainAndTest(adder_inputDim, adder_outputDim, adder_hiddenLayerDim, adder_X_train, adder_X_test, adder_y_train, adder_y_test, "Two Bit Adder", adder_learningRate, numIterations)

elapsedTime = time.time() - start

print("Summary:")
print("Each hyperparameter was tested further than")
print("what is about to be displayed, but testing was")
print("limited to provide a reasonable run time.")
print("Given learning rates: [.01,.1,.5], Number of Epochs: [10,100,1000],")
print("and hidden layers dims [4,8,16];[34,64,128], these were the best parameters:")
print()
print("Best Hyperparameters for XOR:")
print("Hidden Layer Dim:", XOR_hiddenLayerDim)
print("Learning Rate:", XOR_learningRate)
print("Number of Epochs:", 1000)
print("Accuracy:", xorAcc, "| Average Loss:", xorLoss)
print()
print("Best Hyperparameters for Two Bit Adder:")
print("Hidden Layer Dim:", adder_hiddenLayerDim)
print("Learning Rate:", adder_learningRate)
print("Number of Epochs:", 1000)
print("Accuracy:", addAcc, "% | Average Loss:", addLoss)
print("\nElapsed time (minutes):", elapsedTime / 60)

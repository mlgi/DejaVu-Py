import numpy as np
from neuron import Neuron, Weight

class NeuralNetwork:
    # a directed graph containing neurons(nodes) and weights(edges)
    def __init__(self, other=None):
        self.inputs = []
        self.neurons = []
        self.weights = []
        self.outputs = []
        # copy constructor
        if other is not None:
            for neuron in other.neurons:
                newron = self.addNeuron(neuron.type, neuron.activation, neuron.bias)
                newron.backpropError = neuron.backpropError
            for weight in other.weights:
                indexOfPrev = other.neurons.index(weight.prev)
                indexOfNext = other.neurons.index(weight.next)
                nWeight = self.addWeight(self.neurons[indexOfPrev], self.neurons[indexOfNext], weight.value)
                nWeight.dValue = weight.dValue
    # adds a new neuron to the list of neurons and appropriately categorizes it
        # type (string) : type of neuron to create (input, hidden, output)
        # activation (string) : activation type of neuron (leaky_relu, tanh, sigmoid, relu)
        # nBias (number) : the bias for the neuron
    def addNeuron(self, type, activation, nBias=None):
        neuron = None
        if nBias is not None:
            neuron = Neuron(type, activation, nBias)
        else:
            neuron = Neuron(type, activation)
        # add to list of neurons and categorize it
        self.neurons.append(neuron)
        if neuron.type == "input":
            self.inputs.append(neuron)
        elif neuron.type == "output":
            self.outputs.append(neuron)
        return neuron
    # adds a new weight to the list of neurons
        # prev (Neuron) : the neuron on the preceding side of the weight
        # next (Neuron) : the neuron on the succeeding side of the weight
        # weightValue (number) : value for the weight
    def addWeight(self, prev, next, weightValue=None):
        newWeight = Weight(prev, next)
        if weightValue is not None:
            newWeight.value = weightValue
        # add to list of weights
        self.weights.append(newWeight)
        return newWeight
    # makes a prediction and returns
        # intakes (np.matrix) : the vector or matrix of shape(# of inputs, 1) containing input values
        # returns the (np.matrix) of output values
    def Predict(self, intakes):
        if intakes.shape[0] == len(self.inputs):
            for i in range(intakes.shape[0]):
                self.inputs[i].setValue(intakes[i,0])
        outputMatrix = np.matrix(np.empty((len(self.outputs), 1)))
        for i in range(len(self.outputs)):
            outputMatrix[i,0] = self.outputs[i].getValue()
        for neuron in self.neurons:
            neuron.activated = False
        return outputMatrix
    # finds the partial derivative of the squared error function to each weight
        # targets (np.matrix) : the vector or matrix of shape(# of outputs, 1) containing target values
    def backpropagate(self, targets):
        if targets.shape[0] == len(self.outputs):
            for i in range(targets.shape[0]):
                self.outputs[i].target = targets[i,0]

        for weight in self.weights:
            weight.dValue = weight.next.getBackpropError() * weight.prev.value

        for neuron in self.neurons:
            neuron.activated = False

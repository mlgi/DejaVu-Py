import numpy as np

class Weight:
    # an edge in a directed graph
        # contains the preceding and succeding neurons
    def __init__(self, prevNeuron, nextNeuron, weightValue=None):
        self.prev = prevNeuron
        self.next = nextNeuron

        if weightValue is not None:
            self.value = weightValue
        else:
            self.value = np.random.normal(0,1)  # initialize a random weight

        self.dValue = None

        self.next.inputWeights.append(self)
        self.next.inputs.append(self.prev)
        self.prev.outputWeights.append(self)

class Neuron:
    def __init__(self, ntype, nactivation, neuronBias=None):
        self.type = ntype   # "input", "hidden", "output"
        self.activation = nactivation   # "leaky_relu", "tanh", "sigmoid", "relu"
        self.inputWeights = [] # list of input weights
        self.outputWeights = [] # list of output weights
        self.inputs = [] # list of input neurons
        self.activated = False
        self.value = None
        self.dValue = None
        self.backpropError = None
        if self.type == "output":
            self.target = None
        if neuronBias is not None:
            self.bias = neuronBias
        else:
            self.bias = np.random.normal(0,1)

    def activate(self, number):
        if self.activation == "leaky_relu":
            return number if number > 0 else 0.01 * number
        elif self.activation == "tanh":
            return np.tanh(number)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-number))
        elif self.activation == "relu":
            return number if number > 0 else 0

    def d_activate(self, number):
        if self.activation == "leaky_relu":
            return 1 if number > 0 else 0.01
        elif self.activation == "tanh":
            return 1.0 / (np.cosh(number) ** 2)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-number)) * (1.0 - (1.0 / (1.0 + np.exp(-number))))
        elif self.activation == "relu":
            return 1 if number > 0 else 0

    def getValue(self):
        # just return the value if it's been evaluated already
        if ( self.value is None or not self.activated ) and self.type != "input":

            weightArray = np.empty((1, len(self.inputWeights)))
            for i in range(len(self.inputWeights)):
                weightArray[0][i] = self.inputWeights[i].value
            weightMatrix = np.matrix(weightArray)

            inputArray = np.empty((len(self.inputs), 1))
            for i in range(len(self.inputs)):
                inputArray[i][0] = self.inputs[i].getValue()
            inputMatrix = np.matrix(inputArray)

            net = (np.dot(weightMatrix, inputMatrix) + np.matrix([[self.bias]]))[0, 0]

            self.value = self.activate(net)
            self.dValue = self.d_activate(net)

            self.activated = True   # used to keep track of whether or not this neuron has been activated already
        return self.value

    def setValue(self, val):
        if self.type == "input":
            self.value = val

    def getBackpropError(self):
        if self.activated == False:
            if self.type == "output":
                self.backpropError = self.dValue * 2 * (self.value - self.target)
            elif self.type != "input":
                sum = 0
                for outputW in self.outputWeights:
                    sum += outputW.value * outputW.next.getBackpropError()
                #print self.dValue
                #print self.type
                self.backpropError = self.dValue * sum
            self.activated = True

        return self.backpropError

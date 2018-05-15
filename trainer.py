import numpy as np
from neuralnetwork import NeuralNetwork

class Trainer:
    def __init__(self, nbatchSize, noptimizer, **kwargs):
        self.batchSize = nbatchSize
        self.optimizer = noptimizer
        self.dataCount = 0
        self.inputSet = []
        self.targetSet = []
        self.params = kwargs
        self.error = 0
        self.lastVector = None
        self.squaredGradientSum = None
        self.rmsGradient = None
        self.rmsUpdate = None

    def addData(self, inputs, targets):
        self.inputSet.append(inputs)
        self.targetSet.append(targets)
        self.dataCount += 1

    def batchpropagate(self, start, network):
        testInputs = self.inputSet[start]
        targetOutputs = self.targetSet[start]
        testOutputs = network.Predict(testInputs)
        network.backpropagate(targetOutputs)

        sumOfWeights = NeuralNetwork(network)

        totalError = 0.0
        for i in range(targetOutputs.shape[0]):
            totalError += (testOutputs[i,0] - targetOutputs[i, 0]) ** 2

        for i in range(start + 1, start + self.batchSize):
            if i >= self.dataCount:
                break
            testInputs = self.inputSet[i]
            targetOutputs = self.targetSet[i]
            testOutputs = network.Predict(testInputs)
            network.backpropagate(targetOutputs)

            for j in range(len(network.weights)):
                sumOfWeights.weights[j].dValue += network.weights[j].dValue

            for i in range(targetOutputs.shape[0]):
                totalError += (testOutputs[i,0] - targetOutputs[i, 0]) ** 2

        for j in range(len(network.weights)):
            sumOfWeights.weights[j].dValue /= self.batchSize
            network.weights[j].dValue = sumOfWeights.weights[j].dValue

        sumOfWeights = None

        return totalError / self.batchSize

    def Train(self, network):
        if "learningRate" in self.params:
            learningRate = self.params["learningRate"]
        else:
            learningRate = 0.01
        # no optimization
        if self.optimizer == "none":
            # for each batch
            for i in range(0, self.dataCount, self.batchSize):
                # find gradients
                self.error = self.batchpropagate(i, network)
                # update weights
                for weight in network.weights:
                    # change vector is learningRate times gradient
                    v = learningRate * weight.dValue
                    weight.value = weight.value - v
        # momentum optimization
        elif self.optimizer == "momentum":
            if "momentumTerm" in self.params:
                momentumTerm = self.params["momentumTerm"]
            else:
                momentumTerm = 0.9
            # for each batch
            for i in range(0, self.dataCount, self.batchSize):
                # find gradient
                self.error = self.batchpropagate(i, network) # find gradients
                if self.lastVector is None:
                    self.lastVector = [0 for weight in network.weights]
                for i in range(len(network.weights)):
                    # change vector is learningRate times gradient plus lastVector times momentumTerm
                    v = learningRate * network.weights[i].dValue + self.lastVector[i] * momentumTerm
                    network.weights[i].value = network.weights[i].value - v
                    self.lastVector[i] = v
        # nesterov accelerated gradient
        elif self.optimizer == "nesterov":
            if "momentumTerm" in self.params:
                momentumTerm = self.params["momentumTerm"]
            else:
                momentumTerm = 0.9
            # subtract weights by momentumTerm times lastVector
            for i in range(len(network.weights)):
                network.weights[i].value = network.weights[i].value - momentumTerm * self.lastVector[i]
            # for each batch
            for i in range(0, self.dataCount, self.batchSize):
                # find gradients
                self.error = self.batchpropagate(i, network)
                # initialize lastVector
                if self.lastVector is None:
                    self.lastVector = [0 for weight in network.weights]
                # update weights
                for i in range(len(network.weights)):
                    v = learningRate * network.weights[i].dValue + self.lastVector[i] * momentumTerm
                    network.weights[i].value = network.weights[i].value - v
                    self.lastVector[i] = v
        # adaptive gradient
        elif self.optimizer == "adagrad":
            # for each batch
            for i in range(0, self.dataCount, self.batchSize):
                # find gradient
                self.error = self.batchpropagate(i, network)
                # t = 0
                if self.squaredGradientSum is None:
                    self.squaredGradientSum = [0 for weight in network.weights]
                    # update weights
                    for i in range(len(network.weights)):
                        network.weights[i].value = network.weights[i].value - learningRate * network.weights[i].dValue
                        self.squaredGradientSum[i] += network.weights[i].dValue ** 2
                # t > 0
                else:
                    # update weights
                    for i in range(len(network.weights)):
                        adaptedRate = learningRate / np.sqrt(self.squaredGradientSum[i])
                        network.weights[i].value = network.weights[i].value - adaptedRate * network.weights[i].dValue
                        self.squaredGradientSum[i] += network.weights[i].dValue ** 2
        # rmsprop
        elif self.optimizer == "rmsprop":
            if "delta" in self.params:
                delta = self.params["delta"]
            else:
                delta = 0.9
            # for each batch
            for i in range(0, self.dataCount, self.batchSize):
                # find gradient
                self.error = self.batchpropagate(i, network)
                # t = 0
                if self.rmsGradient is None:
                    self.rmsGradient = [0 for weight in network.weights]
                    # update weights
                    for i in range(len(network.weights)):
                        network.weights[i].value = network.weights[i].value - learningRate * network.weights[i].dValue
                        # initialize avg squared gradient
                        self.rmsGradient[i] = network.weights[i].dValue ** 2
                # t > 0
                else:
                    # update weights
                    for i in range(len(network.weights)):
                        # update average squared gradient
                        self.rmsGradient[i] = delta * self.rmsGradient[i] + (1 - delta) * (network.weights[i].dValue ** 2)
                        adaptedRate = learningRate / np.sqrt(self.rmsGradient[i])
                        network.weights[i].value = network.weights[i].value - adaptedRate * network.weights[i].dValue
        # adadelta
        elif self.optimizer == "adadelta":
            if "delta" in self.params:
                delta = self.params["delta"]
            else:
                delta = 0.9
            # for each batch
            for i in range(0, self.dataCount, self.batchSize):
                # find gradient
                self.error = self.batchpropagate(i, network)
                # initialize RMS's
                if self.rmsGradient is None and self.rmsUpdate is None:
                    self.rmsGradient = [0 for weight in network.weights]
                    self.rmsUpdate = [0 for weight in network.weights]

                for i in range(len(network.weights)):
                    # find average squared gradient
                    self.rmsGradient[i] = delta * self.rmsGradient[i] + (1.0 - delta) * (network.weights[i].dValue ** 2)
                    # find adapted rate
                    rate = np.sqrt(self.rmsUpdate[i] + 1e-8) / np.sqrt(self.rmsGradient[i] + 1e-8)
                    # update vector is adapted Rate times gradient
                    update = rate * network.weights[i].dValue
                    # update weight
                    network.weights[i].value -= update
                    # find average squared update
                    self.rmsUpdate[i] = delta * self.rmsUpdate[i] + (1.0 - delta) * (update ** 2)

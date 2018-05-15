import numpy as np
from neuralnetwork import NeuralNetwork
from trainer import Trainer
import matplotlib.pyplot as plt
import pickle

network = pickle.load(open("test.boi", "rb"))
ashketchum = Trainer(4, "adadelta", learningRate=0.05, delta=0.9)

ashketchum.addData(np.matrix([[0], [0]]),np.matrix([[0]]))
ashketchum.addData(np.matrix([[0], [1]]),np.matrix([[0]]))
ashketchum.addData(np.matrix([[1], [0]]),np.matrix([[0]]))
ashketchum.addData(np.matrix([[1], [1]]),np.matrix([[1]]))

errors = []

ashketchum.Train(network)
errors.append(ashketchum.error)

plt.show()
axes = plt.gca()
axes.set_xlim(0, len(errors))
axes.set_ylim(0,errors[0])
line, = axes.plot(range(len(errors)), errors)

while ashketchum.error > 0.025:
    ashketchum.Train(network)
    errors.append(ashketchum.error)
    #print ashketchum.error
    axes.set_xlim(0,len(errors))
    axes.set_ylim(0,max(errors))
    line.set_xdata(range(len(errors)))
    line.set_ydata(errors)
    plt.draw()
    plt.pause(1e-25)




while True:
    inputMatrix = np.matrix([[0] for i in range(len(network.inputs))])
    for i in range(len(network.inputs)):
        inputMatrix[i][0] = float(input("input{0}: ".format(i)))

    print(network.Predict(inputMatrix))

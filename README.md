# DejaVu-Py
A project made as the python version of [my previous endeavor](https://github.com/mlgi/DejaVu) in neural networks. Currently supports the ability to make a simple feed-forward model by identifying the neural network as a directed graph. Each directed edge represents a weight and each node represents a neuron. The NeuralNetwork is then just a set containing the set of weights and the set of neurons.

As with the past project, this is sort of representative of what I comprehend in neural networks enough to code it. Since a lot of machine learning projects are in python I decided to jump ship from my previous solitary experiences in C/C++/C# and learn a new language while I'm at it.

# Features
## Model Creation
Create a blank network using the NeuralNetwork class and add neurons and connections with methods:
 - addNeuron()
 - addWeight()
### Neurons
Neurons supports multiple activation functions and can be initialized with a specific bias if desired. Activation functions include:
 - leaky_relu
 - tanh
 - sigmoid
 - relu
### Weights
Weights hold the weight value when passing the value of a previous node to the next. It also holds the gradient for that weight when the network is backpropagated.
## Training
Use the Trainer() class to create a trainer that trains the network to a certain dataset (what a mouthful). Supports mini-batch, batch, and stochastic gradient descent. Make the batch size the same as the size of training data for full batch GD, or 1 for pure SGD. Optimizers include:
 - no optimizer
 - momentum
 - nesterov accelerated gradient
 - adagrad
 - rmsprop
 - adadelta

# Todo
 - a wiki of sorts
 - add more activation functions
 - add more optimizers
   - [ ] Adam
   - [ ] AdaMax
   - [ ] Nadam
   - [ ] AMSGrad
 - front-end @Gablooblue ;)
 - simplify making layers and connecting them
   - make a addLayer() method
   - make a connect() method that takes two lists of neurons and fully connects one from the other
 - add the ability to add a network to an existing network to be able to create complex structures
 - support recurrent neural network
   - learn backpropagation through time
   - learn LSTMs
 - dataset management
   - importing .csv files
   - learning to read the MNIST dataset
   - saving predicted values given inputs
   - basic data I/O
 - learn more about neural networks
 - loading data into Trainer from file instead of reading the entire file into memory to save RAM
   - maybe this will cause slowdowns? not sure o:
# Currently working on
 - reading up on RNNs and such
   - BPTT :(

# Resources
I'll add more to this list if it's not already included in my [previous reference list](https://github.com/mlgi/DejaVu#resources).

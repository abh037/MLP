# MLP
Self-implemented multi-layer perceptron.
# Usage
Create the net with `my_net = Net(structure = [784, 256, 10])` where `structure` is an array denoting the input size (first item), output size (last item), and hidden layer sizes (middle items). Instantiating the netowrk without specifying a structure will auto-generate a structure when the `train` method is called.
The methods of the class are as follows:

`sigmoid(self, x, der = False)` returns the sigmoid of `x`, or the derivative of the sigmoid of `x` if `der` is true.

`save(self, name)` saves the network to a .npy file with name string `name`.

`load(self, name)` loads a network from a .npy file with name string `name`.

`predict(self, X)` predicts the class of input `X` via a forward pass through the network.

`forward(self, X)` runs `X` through a forward pass through the network, and returns an array of floats between 0 and 1, each denoting the likelihood `X` is the class associated with that index.

`train(self, dataset, labels, epochs = 50, batch_size = 1000)` runs `batch_size` randomly chosen items from `dataset` and associated `labels` through a backwards pass through the network `epochs` times.
# Dependencies
numpy

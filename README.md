# MLP
Self-implemented multi-layer perceptron.
# Usage
Create the net with `my_net = Net(structure = [784, 256, 10])` where `structure` is an array denoting the input size (first item), output size (last item), and hidden layer sizes (middle items). Instantiating the network without specifying a structure (like so: `my_net = Net()`) will auto-generate a structure when the `train` method is called.
The methods of the class are as follows:

`save(self, name)` saves the network to a .npy file with name string `name`.

`load(self, name)` loads a network from a .npy file with name string `name`.

`predict(self, X)` predicts the class of input `X` via a forward pass through the network.

`train(self, dataset, labels, epochs = 50, batch_size = 1000)` runs `batch_size` randomly chosen items from `dataset` and associated `labels` through a backwards pass through the network `epochs` times.

`accuracy(dataset, labels)` returns a float representing the percent of the network's calls to `predict` for `dataset` return a result that matches its given label inm `labels`.
# Dependencies
numpy

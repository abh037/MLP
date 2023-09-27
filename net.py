import numpy as np
from random import choices
from os import listdir
np.seterr(all='ignore')


class Net():

    def __init__(self, structure = []):
        """
        Description:
            The class for a fully connected multi-layer perceptron, with user defined layer counts
            and sizes
        Parameters:
            structure (List)      : the structure is the shape of the MLP, where the first number
                                    in the list is the size of the input layer, the last number
                                    in the list is the size of the output layer (i.e. the number of
                                    classes contained within the dataset on which the MLP is to be
                                    trained), and everything in between represents the sizes of
                                    hidden layers; note that this variable is optional, and an
                                    intelligently generated structure will be generated either upon
                                    loading a network with 'load()', or in the 'train()' method upon
                                    training the network
            
        """
    
        if len(structure) == 0:
            self.input_size = 0
            self.num_classes = 0
            self.structure = []
            self.W = [np.array([])]
            self.b = [np.array([])]
            self.a = [np.array([])]
            self.z = [np.array([])]
            
        elif len(structure) < 3:
            raise Exception("Network structure must have at least 3 layers: [input, hidden, ..., output]")
            
        else:
            self.input_size = structure[0]
            self.num_classes = structure[-1]
            self.structure = structure
            self.W = [np.array([])]
            self.b = [np.array([])]
            self.a = [np.array([])]
            self.z = [np.array([])]
            
            for i in range(1, len(structure)):
                self.W.append(np.random.normal(size = (structure[i], structure[i - 1]), scale = 0.1))
                self.b.append(np.zeros((1, structure[i])))
                self.a.append(np.array([]))
                self.z.append(np.array([]))
    
    def _sigmoid(self, x, der = False):
        """
        Description:
            sigmoid function
        Parameters:
            x (Numpy array)       : array the sigmoid function should be applied to
            der (bool)            : whether the derivative of the sigmoid should be applied
        Returns:
            sigmoid (or derivative of sigmoid) of x
        """
        sig = 1/(1+np.exp(-x))
        if der:
            return sig*(1 - sig)
        else:
            return sig
            
    def _forward(self, X):
        """
        Description:
            the forward propogation algorithm
        Parameters:
            X (Numpy array)       : one instance of data of the same type the network was trained on
        Returns:
            the output of applying the net to X
        """
        self.a[0] = X.flatten().reshape(self.input_size, 1)
        self.z[0] = np.empty((1, self.input_size))

        for i in range(1, len(self.structure)):
            self.z[i] = np.dot(self.W[i], self.a[i - 1]).T + self.b[i]
            self.a[i] = self._sigmoid(self.z[i]).T
        return self.a[-1].T
    
    def _backward(self, X, y, lr):
        """
        Description:
            the backpropogation algorithm, adjusts the weights and biases of the network
        Parameters:
            X (Numpy array)       : one instance of data of the same type the network was trained on
            y (int)               : the label associated with X
            lr (float)            : the learning rate, determines how much weights and biases should
                                    be adjusted by
        Returns:
            the mean square error of y and applying the forward propagation algorithm to X
        """
        dedz = np.subtract(y, self._forward(X))
        error = np.square(dedz).mean()
        
        for i in range(len(self.structure) - 1, 0, -1):
            self.W[i] += lr * dedz.T.dot(self.a[i - 1].T)
            self.b[i] += lr * dedz
            dedz = np.multiply(dedz.dot(self.W[i]), self._sigmoid(self.z[i - 1], der = True))
        return error
    
    def save(self, name):
        """
        Description:
            saves the weights and biases of the MLP to the current directory
        Parameters:
            name (string)         : the name to be associated with the current MLP's set
                                    of weights and biases
        """
        np.save(name + "_W.npy", np.array(self.W, dtype=object), allow_pickle = True)
        np.save(name + "_b.npy", np.array(self.b, dtype=object), allow_pickle = True)
    
    def load(self, name):
        """
        Description:
            loads weights associated with 'name' from the current directory
        Parameters:
            name (string)         : the name associated with the weights and biases of the
                                    MLP you would like to load
        """
        weights = np.load(name + "_W.npy", allow_pickle = True)
        biases = np.load(name + "_b.npy", allow_pickle = True)
        struc = [weights[1].shape[1]]
        for W in weights[1:]:
            struc.append(W.shape[0])
        self.__init__(structure = struc)
        self.W = weights
        self.b = biases
        
    def list_available(self):
        """
        Description:
            lists the valid names of loadable sets of wieghts and biases in the current directory
        """
        for n in {name[:-6] for name in listdir() if ".npy" in name}:
            print(n)

    def predict(self, X):
        """
        Description:
            applies the network to X to predict its associated label
        Parameters:
            X (Numpy array)     : one instance of data of the same type the network was trained on
        Returns:
            the integer associated with predicted label
        """
        if len(self.structure) == 0:
            raise Exception("Can't predict with an untrained network!")
        return np.argmax(self._forward(X))
        
    def accuracy(self, dataset, labels):
        """
        Description:
            outputs the accuracy of the MLP's label predictions for an input dataset
        Parameters:
            dataset (Numpy array)  : set of data of the same type the network was trained on
            labels (Numpy array)   : set of labels associated with 'dataset'
        Returns:
            the accuracy of the network as a float
        """
        avg = 0
        for x, y in zip(dataset, labels):
            if self.predict(x) == y:
                avg += 1
        return 100 * avg/len(dataset)

    def train(self, dataset, labels, epochs = 50, batch_size = 1000, learning_rate = 0.0001):
        """
        Description:
            a handler function to receive data and label, initialize the network if not done
            so already, train the network on the data, and prints the progress of this training
        Parameters:
            dataset (Numpy array)  : set of data of the same type the network was trained on
            labels (Numpy array)   : set of labels associated with 'dataset'
            epochs (int)           : number of times the network should train across a subset
                                     of the input data
            batch_size (int)       : number of stochastically selected elements of 'dataset'
                                     that the network should be trained on each epoch
            learning_rate (float)  : the magnitude by which weights and biases should be adjusted
                                     each itertion of the backpropagation algorithm
        """
    
        if batch_size > len(dataset):
            batch_size = len(dataset)
            
        if len(self.structure) == 0:
            struc = [len(np.array(dataset[0]).flatten()), int(2 * len(np.array(dataset[0]).flatten()) / 3), len(np.unique(labels))]
            self.__init__(structure = struc)
            print("No network structure specified, using auto-generated structure " + str(struc))

        
        if len(np.array(dataset[0]).flatten()) != self.structure[0]:
            raise Exception("Training dataset elements count (" + str(len(np.array(dataset[0]).flatten())) + ") must be the same size as the input layer (" + str(self.structure[0]) + ")")
    
        reshaped_dataset = np.array(dataset)
        reshaped_dataset = reshaped_dataset.flatten().reshape(len(dataset), self.input_size)
        reshaped_labels = np.zeros((len(labels), self.num_classes))
        for i, label in enumerate(labels):
            reshaped_labels[i][label] = 1
            
        for ep in range(epochs):
            
            running_error = 0
            for i, (X, y) in choices(list(enumerate(zip(reshaped_dataset, reshaped_labels))), k = batch_size):
                error = self._backward(X, y, learning_rate)
                running_error += error
            running_error = running_error / batch_size
            
            print(f"Epoch: {ep + 1}/{epochs}, Error: {str(running_error)[:7]}", end = '\r')
        print()
        
        
    


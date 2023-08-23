import numpy as np
from random import choices
np.seterr(all='ignore')


class Net():

    def sigmoid(self, x, der = False):
        sig = 1/(1+np.exp(-x))
        if der:
            return sig*(1 - sig)
        else:
            return sig

    def __init__(self, structure = []):
    
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
    def save(self, name):
        np.save(name + ".npy", np.array([self.W, self.b]), allow_pickle = True)
    
    def load(self, name):
        weights, biases = np.load(name + ".npy", allow_pickle = True)
        struc = [weights[1].shape[1]]
        for W in weights[1:]:
            struc.append(W.shape[0])
        self.__init__(structure = struc)
        self.W = weights
        self.b = biases

    def predict(self, X):
        return np.argmax(self.forward(X))
        
    def accuracy(self, dataset, labels):
        avg = 0
        for x, y in zip(dataset, labels):
            if self.predict(x) == y:
                avg += 1
        return 100 * avg/len(dataset)

    def forward(self, X):
        self.a[0] = X.flatten().reshape(self.input_size, 1)
        self.z[0] = np.empty((1, self.input_size))

        for i in range(1, len(self.structure)):
            self.z[i] = np.dot(self.W[i], self.a[i - 1]).T + self.b[i]
            self.a[i] = self.sigmoid(self.z[i]).T
        return self.a[-1].T

    def train(self, dataset, labels, epochs = 50, batch_size = 1000):
    
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
            
            learning_rate = 0.0001
            
            for i, (X, y) in choices(list(enumerate(zip(reshaped_dataset, reshaped_labels))), k = batch_size):
            
                error = np.square(np.subtract(y, self.forward(X))).mean()
                dedz = np.subtract(y, self.forward(X))
                
                for i in range(len(self.structure) - 1, 0, -1):
                    self.W[i] += learning_rate * dedz.T.dot(self.a[i - 1].T)
                    self.b[i] += learning_rate * dedz
                    dedz = np.multiply(dedz.dot(self.W[i]), self.sigmoid(self.z[i - 1], der = True))
                
                running_error += error
            
            running_error = running_error / batch_size
            
            print("Epoch: " + str(ep + 1) + "/" + str(epochs) + ", Error: " + str(running_error)[:7], end = '\r')
        print()
        
        
    


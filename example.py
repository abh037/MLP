from net import Net
import os
from keras import datasets

(trainx, trainy), (testx, testy) = datasets.mnist.load_data()

my_net = Net()
my_net.train(trainx, trainy)
my_net.save("mnist")

my_other_net = Net()
my_other_net.list_available()
my_other_net.load("mnist")

print(my_other_net.accuracy(testx, testy))


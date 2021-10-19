import sys

import numpy as np
import neuralnetwork as nn
from mnist import MNIST

print("Starting...")

mndata = MNIST('mnist-dataset')


num_classes = 10
net = nn.neural_network(3, [784, 20, 10], [None, "tanh", "softmax"], cost_function="cross_entropy")

def train():
    print("Training...")

    # train
    images, labels = mndata.load_training()
    targets = np.array([labels]).reshape(-1)
    one_hot_targets = np.eye(num_classes)[labels]
    net.train(1, inputs=images, labels=one_hot_targets, num_epochs=1, learning_rate=0.001, filename="savepoint.pkl")
    return()

def test():
    print("Testing...")

    # test
    images, labels = mndata.load_testing()
    targets = np.array([labels]).reshape(-1)
    one_hot_targets = np.eye(num_classes)[labels]
    net.check_accuracy("savepoint.pkl", images, one_hot_targets)

    return()


if __name__=='__main__':
    while True:
        if input('Enter 0 for Training , 1 for testing') == '1':
            test()
            sys.exit()
        else:
            train()

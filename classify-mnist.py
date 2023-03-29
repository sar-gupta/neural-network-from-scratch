
import numpy as np
import neuralnetwork as nn
#

print("Starting...")

from mnist import MNIST


def tester(netwotk, mndatan, num_classesn):
    print("Testing...")

    # test
    images, labels = mndata.load_testing()
    targets = np.array([labels]).reshape(-1)
    one_hot_targets = np.eye(num_classes)[labels]
    net.check_accuracy("savepoint.pkl", images, targets)

    return ()


def trainer(network, mndatan, num_classesn):
    print("Training...")

    # train
    images, labels = mndata.load_training()
    targets = np.array([labels]).reshape(-1)
    one_hot_targets = np.eye(num_classes)[labels]
    net.train(1, inputs=images, labels=one_hot_targets, num_epochs=1, learning_rate=0.001,
              filename="savepoint.pkl")
    return ()


while True:

        if input('Enter 0 for Training , 1 for testing') == '1':
            mndata = MNIST('mnist-dataset')

            num_classes = 10
            net = nn.neural_network(3, [784, 20, 10], activation_function=nn.sigmoid, cost_function=nn.cross_entropy)

            tester(net, mndata, num_classes)
            #sys.exit()
        else:
            mndata = MNIST('mnist-dataset')

            num_classes = 10
            net = nn.neural_network(3, [784, 20, 10], activation_function=nn.sigmoid, cost_function=nn.cross_entropy)

            trainer(net,mndata,  num_classes)

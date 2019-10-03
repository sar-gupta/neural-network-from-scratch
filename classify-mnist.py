import numpy as np
from mnist import MNIST

import neuralnetwork as nn

print("Starting...")

mndata = MNIST("mnist-dataset")
images, labels = mndata.load_training()

print("Training...")

# train
num_classes = 10
targets = np.array([labels]).reshape(-1)
one_hot_targets = np.eye(num_classes)[labels]
net = nn.neural_network(
    3, [784, 20, 10], [None, "tanh", "softmax"], cost_function="cross_entropy"
)
net.train(
    1,
    inputs=images,
    labels=one_hot_targets,
    num_epochs=5,
    learning_rate=0.001,
    filename="savepoint.pkl",
)

print("Testing...")

# test
images, labels = mndata.load_testing()
targets = np.array([labels]).reshape(-1)
one_hot_targets = np.eye(num_classes)[labels]
net.check_accuracy("savepoint.pkl", images, one_hot_targets)

import dill
import numpy as np


class neural_network:
    def __init__(self, num_layers, num_nodes, activation_function, cost_function):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.layers = []
        self.cost_function = cost_function

        if not num_layers == len(num_nodes):
            raise ValueError("Number of layers must match number node counts")

        for i in range(num_layers):
            if i != num_layers - 1:
                layer_i = layer(num_nodes[i], num_nodes[i + 1], activation_function[i])
            else:
                layer_i = layer(num_nodes[i], 0, activation_function[i])
            self.layers.append(layer_i)

    def check_training_data(self, batch_size, inputs, labels):
        self.batch_size = batch_size
        if not len(inputs) % self.batch_size == 0:
            raise ValueError("Batch size must be multiple of number of inputs")
        if not len(inputs) == len(labels):
            raise ValueError("Number of inputs must match number of labels")
        for i in range(len(inputs)):
            if not len(inputs[i]) == self.num_nodes[0]:
                raise ValueError(
                    "Length of each input data must match number of input nodes"
                )
            if not len(labels[i]) == self.num_nodes[-1]:
                raise ValueError(
                    "Length of each label data must match number of output nodes"
                )

    def train(self, batch_size, inputs, labels, num_epochs, learning_rate, filename):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.check_training_data(self.batch_size, inputs, labels)
        for j in range(num_epochs):
            i = 0
            print("== EPOCH: ", j + 1, "/", num_epochs, " ==")
            while i + batch_size != len(inputs):
                print("Training with ", i + batch_size + 1, "/", len(inputs), end="\r")
                self.error = 0
                self.forward_pass(inputs[i : i + batch_size])
                self.calculate_error(labels[i : i + batch_size])
                self.back_pass(labels[i : i + batch_size])
                i += batch_size
            self.error /= batch_size
            print("\nError: ", self.error)
        print("Saving...")
        dill.dump_session(filename)

    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers - 1):
            temp = np.add(
                np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer),
                self.layers[i].bias_for_layer,
            )
            if self.layers[i + 1].activation_function == "sigmoid":
                self.layers[i + 1].activations = self.sigmoid(temp)
            elif self.layers[i + 1].activation_function == "softmax":
                self.layers[i + 1].activations = self.softmax(temp)
            elif self.layers[i + 1].activation_function == "relu":
                self.layers[i + 1].activations = self.relu(temp)
            elif self.layers[i + 1].activation_function == "tanh":
                self.layers[i + 1].activations = self.tanh(temp)
            else:
                self.layers[i + 1].activations = temp

    def relu(self, layer):
        layer[layer < 0] = 0
        return layer

    def softmax(self, layer):
        exp = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            return exp / np.sum(exp, axis=1, keepdims=True)
        else:
            return exp / np.sum(exp, keepdims=True)

    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(np.negative(layer))))

    def tanh(self, layer):
        return np.tanh(layer)

    def calculate_error(self, labels):
        if len(labels[0]) != self.layers[self.num_layers - 1].num_nodes_in_layer:
            print("Error: Label is not of the same shape as output layer.")
            print("Label: ", len(labels), " : ", len(labels[0]))
            print(
                "Out: ",
                len(self.layers[self.num_layers - 1].activations),
                " : ",
                len(self.layers[self.num_layers - 1].activations[0]),
            )
            return

        if self.cost_function == "mean_squared":
            self.error += np.mean(
                np.divide(
                    np.square(
                        np.subtract(
                            labels, self.layers[self.num_layers - 1].activations
                        )
                    ),
                    2,
                )
            )
        elif self.cost_function == "cross_entropy":
            self.error += np.negative(
                np.sum(
                    np.multiply(
                        labels, np.log(self.layers[self.num_layers - 1].activations)
                    )
                )
            )

    def back_pass(self, labels):
        # if self.cost_function == "cross_entropy" and self.layers[self.num_layers-1].activation_function == "softmax":
        targets = labels
        i = self.num_layers - 1
        y = self.layers[i].activations
        deltab = np.multiply(y, np.multiply(1 - y, targets - y))
        deltaw = np.matmul(np.asarray(self.layers[i - 1].activations).T, deltab)
        new_weights = self.layers[i - 1].weights_for_layer - self.learning_rate * deltaw
        new_bias = self.layers[i - 1].bias_for_layer - self.learning_rate * deltab
        for i in range(i - 1, 0, -1):
            y = self.layers[i].activations
            deltab = np.multiply(
                y,
                np.multiply(
                    1 - y,
                    np.sum(np.multiply(new_bias, self.layers[i].bias_for_layer)).T,
                ),
            )
            deltaw = np.matmul(
                np.asarray(self.layers[i - 1].activations).T,
                np.multiply(
                    y,
                    np.multiply(
                        1 - y,
                        np.sum(
                            np.multiply(new_weights, self.layers[i].weights_for_layer),
                            axis=1,
                        ).T,
                    ),
                ),
            )
            self.layers[i].weights_for_layer = new_weights
            self.layers[i].bias_for_layer = new_bias
            new_weights = (
                self.layers[i - 1].weights_for_layer - self.learning_rate * deltaw
            )
            new_bias = self.layers[i - 1].bias_for_layer - self.learning_rate * deltab
        self.layers[0].weights_for_layer = new_weights
        self.layers[0].bias_for_layer = new_bias

    def predict(self, filename, input):
        dill.load_session(filename)
        self.batch_size = 1
        self.forward_pass(input)
        a = self.layers[self.num_layers - 1].activations
        a[np.where(a == np.max(a))] = 1
        a[np.where(a != np.max(a))] = 0
        return a

    def check_accuracy(self, filename, inputs, labels):
        dill.load_session(filename)
        self.batch_size = len(inputs)
        self.forward_pass(inputs)
        a = self.layers[self.num_layers - 1].activations
        a[np.where(a == np.max(a))] = 1
        a[np.where(a != np.max(a))] = 0
        total = 0
        correct = 0
        for i in range(len(a)):
            total += 1
            if np.equal(a[i], labels[i]).all():
                correct += 1
        print("Accuracy: ", correct * 100 / total)

    def load_model(self, filename):
        dill.load_session(filename)


class layer:
    def __init__(
        self, num_nodes_in_layer, num_nodes_in_next_layer, activation_function
    ):
        self.num_nodes_in_layer = num_nodes_in_layer
        self.activation_function = activation_function
        self.activations = np.zeros([num_nodes_in_layer, 1])
        if num_nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(
                0, 0.001, size=(num_nodes_in_layer, num_nodes_in_next_layer)
            )
            self.bias_for_layer = np.random.normal(
                0, 0.001, size=(1, num_nodes_in_next_layer)
            )
        else:
            self.weights_for_layer = None
            self.bias_for_layer = None

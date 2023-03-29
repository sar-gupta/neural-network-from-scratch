# a feed forawrd network with random weights
# the weights are updated using backpropagation

import numpy as np
import dill


def sigmoid(layer):
    return np.divide(1, np.add(1, np.exp(np.negative(layer))))


def softmax(layer):
    exps = np.exp(layer)
    return exps / np.sum(exps)


def relu(layer):
    return np.maximum(0, layer)


def tanh(layer):
    return np.tanh(layer)


def leaky_relu(layer):
    return np.maximum(0.01 * layer, layer)


def mean_squared(labels, layers, num_layers):
    return np.mean(np.divide(np.square(np.subtract(labels, layers[num_layers - 1].activations)), 2))


def cross_entropy(layers, lables, num_layers):
    return np.negative(np.sum(np.multiply(lables, np.log(layers[num_layers - 1].activations))))


class layer:
    def __init__(self, num_nodes_in_layer, num_nodes_in_next_layer):
        self.num_nodes_in_layer = num_nodes_in_layer
        # self.activation_function = activation_function
        self.activations = np.zeros((num_nodes_in_layer, 1))
        if num_nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(num_nodes_in_layer, num_nodes_in_next_layer))
            self.bias_for_layer = np.random.normal(0, 0.001, size=(1, num_nodes_in_next_layer))
        else:
            self.weights_for_layer = None
            self.bias_for_layer = None


class neural_network:

    def __init__(self, num_layers, num_nodes, activation_function=sigmoid, cost_function=mean_squared):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.layers = []
        self.cost_function = cost_function
        self.activation_function = activation_function

        if num_layers != len(num_nodes):
            raise ValueError("Number of layers must match number node counts")

        num_count = num_layers
        while num_count > 1:
            layer_i = layer(num_nodes[num_layers - num_count], num_nodes[num_layers - num_count + 1])
            self.layers.append(layer_i)
            num_count -= 1
        self.layers.append(layer(num_nodes[num_layers - 1], 0))

    def check_training_data(self, batch_size, inputs, labels):
        # labels=[[a,b,c..],[c,d,e...].....] is the training output for each input. It is technically a matrix
        # self.batch_size = batch_size
        if (len(inputs) % batch_size) != 0:
            raise ValueError("Inputs must be multiple of number of batch size")
        if len(inputs) != len(labels):
            raise ValueError("Number of inputs must match number of labels")
        a = self.num_nodes[0]
        b = self.num_nodes[-1]
        # Just Checking if the input and output are of the same shape
        for i in range(len(inputs)):
            try:
                inputs[i][a]
                print(labels[i][b])
                raise ValueError(f" A Inconsistency in data no. {i} : Inputs : {inputs[i]}, labels :{labels[i]} ")
            except:
                pass
            try:
                inputs[i][a - 1]
                labels[i][b - 1]
            except:
                raise ValueError(f"Inconsistency in data no. {i} : Inputs : {inputs[i]}, labels :{labels[i]} ")

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
                self.forward_pass(inputs[i:i + batch_size])
                self.calculate_error(labels[i:i + batch_size])
                self.back_pass(labels[i:i + batch_size])
                i += batch_size
            self.error /= batch_size
            print("\nError: ", self.error)
        print("Saving...")
        dill.dump_session(filename)

    # def sigmoid(self,layer):
    # return np.divide(1, np.add(1, np.exp(np.negative(layer))))

    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers - 1):
            temp = np.add(np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer),
                          self.layers[i].bias_for_layer)
            # print(type(self.layers[i+1].activations),type(temp))
            self.layers[i + 1].activations = sigmoid(temp)

    def calculate_error(self, labels):
        if len(labels[0]) != self.layers[self.num_layers - 1].num_nodes_in_layer:
            print("Error: Label is not of the same shape as output layer.")
            print("Label: ", len(labels), " : ", len(labels[0]))
            print("Out: ", len(self.layers[self.num_layers - 1].activations), " : ",
                  len(self.layers[self.num_layers - 1].activations[0]))
            return

        self.error += self.cost_function(self.layers, labels, self.num_layers)

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
            deltab = np.multiply(y, np.multiply(1 - y, np.sum(np.multiply(new_bias, self.layers[i].bias_for_layer)).T))
            deltaw = np.matmul(np.asarray(self.layers[i - 1].activations).T, np.multiply(y, np.multiply(1 - y, np.sum(
                np.multiply(new_weights, self.layers[i].weights_for_layer), axis=1).T)))
            self.layers[i].weights_for_layer = new_weights
            self.layers[i].bias_for_layer = new_bias
            new_weights = self.layers[i - 1].weights_for_layer - self.learning_rate * deltaw
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
        print("Accuracy: ", ((correct / total) * 100))

    def load_model(self, filename):
        dill.load_session(filename)

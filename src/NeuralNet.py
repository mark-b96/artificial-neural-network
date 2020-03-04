"""
Author: Mark Bonney

"""
import numpy as np
from Layer import Layer


class NeuralNet(object):
    def __init__(self, l):
        self.learning_rate = l
        self.input_layer, self.output_layer = None, None
        self.num_inputs, self.num_outputs = 0, 0
        self.loss = None
        self.hidden_layers = []  # List of layer objects
        self.bias_neuron = 1  # Set to 1 for all layers

    def read_input_data(self, inputs, targets):
        """
        Read input data and populate input and output layers
        :param inputs: Input data to the model
        :param targets: Desired targets
        """
        self.input_layer = np.loadtxt(inputs, delimiter=",", ndmin=2)
        self.output_layer = np.loadtxt(targets, delimiter=",", ndmin=2)
        self.num_inputs = self.input_layer.shape[0]
        self.num_outputs = self.output_layer.shape[0]

    def initialise_weights(self):
        """Randomly initialise weight matrices for each layer in ANN"""
        np.random.seed(42)
        self.add_layer(self.num_outputs)
        previous_layer = self.hidden_layers[0].num_neurons
        self.hidden_layers[0].output = self.input_layer
        for layer in self.hidden_layers[1:]:
            layer.weights = np.random.rand(layer.num_neurons, previous_layer)
            previous_layer = layer.num_neurons

    def add_layer(self, num_neurons):
        """
        Add a hidden layer to the ANN
        :param num_neurons: Number of neurons in the layer
        """
        new_layer = Layer(num_neurons)
        self.hidden_layers.append(new_layer)

    def forward_pass(self):
        """
        Compute forward pass using matrices
        :return: Output matrix of ANN
        """
        previous_layer = self.input_layer
        for layer in self.hidden_layers[1:]:
            layer.output = self.sigmoid_function(np.dot(layer.weights, previous_layer) + self.bias_neuron)
            previous_layer = layer.output

        self.loss = self.output_layer - self.hidden_layers[-1].output
        return self.hidden_layers[-1].output

    @staticmethod
    def sigmoid_function(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def gradient_sigmoid(f):
        return (1-f)*f

    def back_propagation(self):
        """Compute gradient of all functions and perform back propagation"""
        # Calculate delta at the output layer, using the loss
        self.hidden_layers[-1].deltas = np.array(self.loss * self.gradient_sigmoid(self.hidden_layers[-1].output))
        previous_layer = self.hidden_layers[-1]
        #  Start at last hidden layer and loop through all layers
        for layer in self.hidden_layers[::-1][1:]:
            layer.output_gradient = self.gradient_sigmoid(layer.output)
            layer.deltas = np.dot(previous_layer.weights.T, previous_layer.deltas)*layer.output_gradient
            update = np.dot(layer.output, previous_layer.deltas.T)
            previous_layer.weights += update.T * self.learning_rate
            previous_layer = layer


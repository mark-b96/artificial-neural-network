"""
Author: Mark Bonney

"""
import numpy as np


class NeuralNet(object):
    def __init__(self, l, h, n):
        self.learning_rate = l
        self.num_hidden_layers = h
        self.num_hidden_neurons = n
        self.input_layer, self.hidden_layer, self.output_layer = None, None, None
        self.input_weights, self.output_weights = None, None
        self.num_inputs, self.num_outputs = 0, 0
        self.bias_1, self.bias_2 = 1, 1
        self.predicted_output = None
        self.loss = None

    def read_input_data(self, inputs, targets):
        """
        Read input data and populate input and output layers
        :param inputs: Input data to the model
        :param targets: Desired targets
        :return: None
        """
        self.input_layer = np.loadtxt(inputs, delimiter=",", ndmin=2)
        self.output_layer = np.loadtxt(targets, delimiter=",", ndmin=2)
        self.num_inputs = self.input_layer.shape[0]
        self.num_outputs = self.output_layer.shape[0]

    def initialise_weights(self):
        """Randomly initialise weight matrices"""
        np.random.seed(42)
        self.input_weights = np.random.rand(self.num_hidden_neurons, self.num_inputs)
        self.output_weights = np.random.rand(self.num_outputs, self.num_hidden_neurons)

    def forward_pass(self):
        """Compute forward pass"""
        self.hidden_layer = self.sigmoid_function(np.dot(self.input_weights, self.input_layer) + self.bias_1)
        self.predicted_output = self.sigmoid_function(np.dot(self.output_weights, self.hidden_layer) + self.bias_2)
        self.loss = self.output_layer-self.predicted_output

    @staticmethod
    def sigmoid_function(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def gradient_sigmoid(f):
        return (1-f)*f

    def back_propagation(self):
        """Compute gradient of all functions"""
        delta_output = np.array(self.loss * self.gradient_sigmoid(self.predicted_output))
        delta_h = np.array(self.gradient_sigmoid(self.hidden_layer))
        tmp_hidden_deltas = np.dot(self.output_weights.T, delta_output)
        final_hidden_deltas = tmp_hidden_deltas * delta_h

        update_1 = self.update_weights(self.hidden_layer, delta_output)
        self.output_weights += update_1.T

        update_2 = self.update_weights(self.input_layer, final_hidden_deltas)
        self.input_weights += update_2.T

    def update_weights(self, layer_1, layer_2):
        """
        Use the chain rule to update weights between two layers
        :param layer_1: Inner layer
        :param layer_2: Outer layer
        :return: The update required scaled by the predefined learning rate
        """
        return np.dot(layer_1, layer_2.T)*self.learning_rate


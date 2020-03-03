"""
Author: Mark Bonney

"""
import numpy as np


class NeuralNet(object):
    def __init__(self, l, h, n):
        self.learning_rate = l
        self.input_layer = None
        self.output_layer = None
        self.hidden_layer = np.array([[]])
        self.input_weights = None
        self.output_weights = None
        self.num_inputs, self.num_outputs = 0, 0
        self.num_hidden_layers = h
        self.num_hidden_neurons = n
        self.b1 = 1
        self.b2 = 1
        self.h, self.delta_h = None, None
        self.output, self.delta_output = None, None
        self.loss = None

    def read_input_data(self, inputs, targets):
        """Read input data and populate input and output layers"""
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
        self.h = self.sigmoid_function(np.dot(self.input_weights, self.input_layer) + self.b1)
        self.output = self.sigmoid_function(np.dot(self.output_weights, self.h) + self.b2)

        self.loss = self.output_layer-self.output

    def sigmoid_function(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def gradient_sigmoid(self, f):
        return (1-f)*f

    def back_propagation(self):
        """Compute gradient of all functions"""

        delta_output = np.array(self.loss * self.gradient_sigmoid(self.output))
        delta_h = np.array(self.gradient_sigmoid(self.h))
        tmp_hidden_deltas = np.dot(self.output_weights.T, delta_output)
        final_hidden_deltas = tmp_hidden_deltas * delta_h

        update_1 = np.dot(self.h, delta_output.T)*self.learning_rate
        self.output_weights += update_1.T

        update_2 = np.dot(self.input_layer, final_hidden_deltas.T)*self.learning_rate
        self.input_weights += update_2.T

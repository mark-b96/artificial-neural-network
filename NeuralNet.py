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
        self.h, self.d_h = None, None
        self.output, self.d_output = None, None
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
        self.output = np.dot(self.output_weights, self.h) + self.b2

        self.loss = self.output - self.output_layer
        print(self.loss)

    def sigmoid_function(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def gradient_sigmoid(self, f):
        return (1-f)*f

    def back_propagation(self):
        """Compute gradient of all functions"""

        d_o = np.array(self.loss * self.gradient_sigmoid(self.loss))
        d_h = np.array(self.gradient_sigmoid(self.h))
        # print(d_h)
        print(d_o.shape)
        self.d_output = np.dot(d_o, self.output.T)
        self.output_weights += -self.learning_rate*self.d_output

        self.d_h = np.dot(d_h, self.h.T)
        self.input_weights += -self.learning_rate*self.d_h

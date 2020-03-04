"""
Author: Mark Bonney

"""


class Layer(object):
    def __init__(self, n):
        self.num_neurons = n
        self.weights = None
        self.output = None
        self.output_gradient = None
        self.deltas = None

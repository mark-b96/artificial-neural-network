"""
Author: Mark Bonney

"""
from NeuralNet import NeuralNet
import numpy as np

if __name__ == '__main__':
    learning_rate = 0.1
    epochs = 10_000
    hidden_layers = 1
    hidden_neurons = 5
    inputs = 'Datasets/q3TrainInputs.csv'
    targets = 'Datasets/q3TrainTargets.csv'
    nn = NeuralNet(learning_rate, hidden_layers, hidden_neurons)
    nn.read_input_data(inputs, targets)
    nn.initialise_weights()
    for epoch in range(epochs):
        nn.forward_pass()
        nn.back_propagation()

    inputs = 'Datasets/q3TestInputs.csv'
    targets = 'Datasets/q3TestTargets.csv'
    nn.read_input_data(inputs, targets)
    nn.forward_pass()
    print(np.round(nn.output))

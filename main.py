"""
Author: Mark Bonney

"""
from NeuralNet import NeuralNet


if __name__ == '__main__':
    learning_rate = 0.01
    epochs = 10
    hidden_layers = 1
    hidden_neurons = 2
    inputs = 'Datasets/q2inputs.csv'
    targets = 'Datasets/q2targets.csv'
    nn = NeuralNet(learning_rate, hidden_layers, hidden_neurons)
    nn.read_input_data(inputs, targets)
    nn.initialise_weights()
    # for epoch in range(epochs):
    nn.forward_pass()
    nn.back_propagation()

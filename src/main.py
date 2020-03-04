"""
Author: Mark Bonney

"""
from NeuralNet import NeuralNet
import numpy as np
import time


def train_model(num_epochs):
    print('Training...')
    start_time = time.time()
    for epoch in range(num_epochs):
        nn.forward_pass()
        nn.back_propagation()
    end_time = time.time()-start_time
    print('Training completed in: {} seconds'.format(end_time))


def test_model():
    input_data = 'Datasets/q3TestInputs.csv'
    target_data = 'Datasets/q3TestTargets.csv'
    nn.read_input_data(input_data, target_data)
    print('\nError matrix')
    print(np.round(nn.forward_pass())-nn.output_layer)


if __name__ == '__main__':
    learning_rate = 0.1
    epochs = 10_000
    input_data = 'Datasets/q3TrainInputs.csv'
    target_data = 'Datasets/q3TrainTargets.csv'
    nn = NeuralNet(learning_rate)
    nn.read_input_data(input_data, target_data)
    nn.add_layer(nn.num_inputs)
    nn.add_layer(4)
    nn.add_layer(4)
    nn.add_layer(4)
    nn.initialise_weights()

    train_model(epochs)
    test_model()




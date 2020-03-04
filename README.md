# Artificial Neural Network
## Multi-layer ANN implemented from first principles
The learning rate and number of epochs can be defined in the main class.

A hidden layer can be added to the ANN using the following code. The argument
specifies the number of neurons in the hidden layer.

```python
nn.add_layer(4)
```

Specify the input and target datasets by modifying the csv input files.

```python
input_data = 'Datasets/q3TrainInputs.csv'
target_data = 'Datasets/q3TrainTargets.csv'
```


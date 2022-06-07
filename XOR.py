from unnati_deeplearning_lib.nn import NeuralNet
from unnati_deeplearning_lib.layers import Linear, Tanh
from unnati_deeplearning_lib.train import train
import numpy as np


def main():
    # Create the neural network.
    nn = NeuralNet([
        Linear(2, 2),
        Tanh(),
        Linear(2, 1),
    ])

    # Create the training data.
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    targets = np.array([
        [0],
        [1],
        [1],
        [0],
    ])

    # Train the neural network.
    train(nn, inputs, targets, num_epochs=500)

    # Make some predictions.
    preds = nn.forward(inputs)
    print("this is printing just fine:", preds)

main()
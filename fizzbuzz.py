from unnati_deeplearning_lib.nn import NeuralNet
from unnati_deeplearning_lib.layers import Tanh, Linear
from unnati_deeplearning_lib.train import train
import numpy as np

def encode_int(x):
    return [1 if i == x else 0 for i in range(10)]
    


def fizzbuzz(x):
    if x % 15 == 0:
        return [1, 0, 0, 0]
    
    if x % 3 == 0:
        return [0, 1, 0, 0]

    if x % 5 == 0:
        return [0, 0, 1, 0]

    return [0, 0, 0, 1]

def __main__():

    # Create a Neural Network
    nn = NeuralNet(
        layers=[
            Linear(input_size=1, output_size=5),
            Tanh(),
            Linear(input_size=5, output_size=4)
        ]     
    )

    # Train the Neural Network
    train(nn, 
        inputs=np.array([i for i in range(1, 10)]),
        targets=np.array([fizzbuzz(i) for i in range(1, 10)]),
        num_epochs=1000,
    )

    # Make some predictions
    preds = nn.forward(inputs= np.array([i for i in range(101, 200)]))

    print(preds)

__main__()
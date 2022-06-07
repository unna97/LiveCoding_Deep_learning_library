'''
A loss function is a function that measures the difference between predictions and actual values.
i.e how good our model is at predicting the correct values.

'''
from unnati_deeplearning_lib.tensor import Tensor
import numpy as np

class Loss:

    def loss(self, actual: Tensor, predicted: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, actual: Tensor, predicted: Tensor) -> Tensor:
        raise NotImplementedError
    
    
class MSE(Loss):
    '''
        MSE is mean squared error
    '''
    def loss(self, actual: Tensor, predicted: Tensor) -> float:
        return np.mean((actual - predicted)**2)

    def grad(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class TSE(Loss):
    '''
        TSE is total squared error
    '''
    def loss(self, actual: Tensor, predicted: Tensor) -> float:
        return np.sum((actual - predicted)**2)
    
    def grad(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return 2 * (predicted - actual) 

class cross_entropy(Loss):
    '''
        cross_entropy is the loss function for classification problems
    '''
    def loss(self, actual: Tensor, predicted: Tensor) -> float:
        return np.sum(-actual * np.log(predicted))

    def grad(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return predicted - actual
'''
Our neural network will be made up of layers. Each layer will be a class.
Each layer needs to propgate it's gradients backwards and pass it's inputs forward.
for example:
    input => linear => tanh => linear => output
'''
from typing import Callable, Dict
import numpy as np
from unnati_deeplearning_lib.tensor import Tensor

class Layer:

    def __init__(self) -> None:
        self.params:Dict[str, Tensor] = {}
        self.grads:Dict[str, Tensor] = {}
        
    
    def forward(self, inputs: Tensor) -> Tensor:
        '''
        Produce the output of the layer given the inputs.
        
        '''
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        '''
        Given the gradient of the loss with respect to the output of the layer,
        update the gradients of the loss with respect to the layer's inputs.
        Return the gradient of the loss with respect to the layer's inputs.
        '''
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, input_size:int, output_size:int) -> None:
        '''
        Initializes the layer.
        input = (batch_size, input_size)
        output = (batch_size, output_size)
        '''

        super().__init__()
        self.params['weight'] = np.random.randn(input_size, output_size)
        self.params['bias'] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        '''
        Produce the output of the layer given the inputs.
        '''
        self.inputs = inputs
        return inputs.dot(self.params['weight']) + self.params['bias']

    def backward(self, grad: Tensor) -> Tensor:
        '''
        Given the gradient of the loss with respect to the output of the layer,
        update the gradients of the loss with respect to the layer's inputs.
        Return the gradient of the loss with respect to the layer's inputs.
    
        '''
        self.grads['weight'] = self.inputs.T @ grad
        self.grads['bias'] = grad.sum(axis=0)

        return grad @ self.params['weight'].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    '''
    An activation layer is a layer that applies a function to 
    it's inputs elementwise i.e 1 to 1 mapping
    '''
    def __init__(self, f: F, f_prime: F) -> None:

        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad 
    


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)        

def tanh_prime(x: Tensor) -> Tensor:
    return 1 - np.tanh(x)**2

class Tanh(Activation):

    def __init__(self) -> None:
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x)**2)

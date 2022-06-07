

'''

'''
from unnati_deeplearning_lib.nn import NeuralNet

class Optimizer:
    
    def __init__(self) -> None:
        pass

    def step (self, net: NeuralNet) -> None:
        raise NotImplementedError
    
class SGD(Optimizer):

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        ##theta = theta - lr * grad
        
    def step(self, net: NeuralNet) -> None:
        for params, grad in net.get_params_and_grads():
            params-= self.lr * grad
    
from unnati_deeplearning_lib.optim import Optimizer, SGD
from unnati_deeplearning_lib.tensor import Tensor
from unnati_deeplearning_lib.loss import TSE, Loss
from unnati_deeplearning_lib.data import DataIterator, BatchIterator
from unnati_deeplearning_lib.nn import NeuralNet



def train(nn: NeuralNet,
        inputs: Tensor,
        targets: Tensor,
        optim: Optimizer = SGD(),
        iterator: DataIterator = BatchIterator(),
        loss: Loss=TSE(),
        num_epochs: int = 10,
        log_every: int = 1
        ) -> None:
    '''
    Train a neural network with the given parameters.
    
    Args:
        nn: A neural network.
        optim: An optimizer.
        loss: A loss function.
        num_epochs: Number of epochs to train for.
        log_every: Print loss after this many epochs.
    '''
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in iterator(inputs, targets):
            predicted = nn.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            nn.backward(grad)
            optim.step(nn)
        
        if epoch%log_every == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss))


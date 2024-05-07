from rsdl.optim import Optimizer
from .. import Tensor
# TODO: implement step function
class SGD(Optimizer):
    def __init__(self, layers, learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            l.weight = l.weight - l.weight.grad * Tensor([self.learning_rate])
            if l.need_bias:
                l.bias = l.bias - l.bias.grad * Tensor([self.learning_rate])
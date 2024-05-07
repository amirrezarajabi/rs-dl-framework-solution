from rsdl import Tensor, Dependency
import numpy as np
from typing import List


def Sigmoid(t: Tensor) -> Tensor:
    # TODO: implement sigmoid function
    # hint: you can do it using function you've implemented (not directly define grad func)
    shape = t.shape
    numerator = Tensor(np.ones(shape))
    denominator = Tensor(np.ones(shape)) + ((-t).exp())
    denominator = denominator**-1
    return numerator * denominator


def Tanh(t: Tensor) -> Tensor:
    # TODO: implement tanh function
    # hint: you can do it using function you've implemented (not directly define grad func)
    numerator = t.exp() - (-t).exp()
    denominator = t.exp() + (-t).exp()
    denominator = denominator**-1
    return numerator * denominator


def Softmax(t: Tensor) -> Tensor:
    # TODO: implement softmax function
    # hint: you can do it using function you've implemented (not directly define grad func)
    # hint: you can't use sum because it has not axis argument so there are 2 ways:
    #        1. implement sum by axis
    #        2. using matrix mul to do it :) (recommended)
    # hint: a/b = a*(b^-1)

    _ones = np.ones((t.shape[-1], 1))
    _t = t.exp()
    sum = _t @ _ones
    sum = sum**-1
    return _t * sum


def Relu(t: Tensor) -> Tensor:
    # TODO: implement relu function

    # use np.maximum
    data = np.where(t.data > 0, t.data, 0)

    req_grad = t.requires_grad
    if req_grad:

        def grad_fn(grad: np.ndarray):
            return np.where(t.data > 0, grad, 0)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def LeakyRelu(t: Tensor, leak=0.05) -> Tensor:
    """
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """
    # TODO: implement leaky_relu function

    data = np.where(t.data > 0, t.data, leak * t.data)

    req_grad = t.requires_grad
    if req_grad:

        def grad_fn(grad: np.ndarray):
            return np.where(t.data > 0, grad, grad * leak)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

from rsdl import Tensor
from ..activations import Softmax
import numpy as np


def MeanSquaredError(preds: Tensor, actual: Tensor):
    # TODO : implement mean squared error
    error = preds - actual
    error2 = error**2
    mse = error2
    size = Tensor(np.array([error2.data.size],dtype=np.float64))
    size = size**-1
    return mse * size


def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    _preds = Softmax(preds)
    _sum = (actual * _preds).sum()
    size = Tensor(np.ndarray(preds.shape).fill(actual.shape[0]))
    size = size**-1
    return _sum * size

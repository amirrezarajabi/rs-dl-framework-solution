# Task 4
import numpy as np
import torch
from tqdm import tqdm as tq
from rsdl import Tensor
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 6

# TODO: define w and b (y = w x + b) with random initialization ( you can use np.random.randn )
w = Tensor(np.random.randn(3,), requires_grad=True)
b = Tensor(np.random.randn(1,), requires_grad=True)

print(w)
print(b)

learning_rate = 0.01
batch_size = 20

for epoch in range(100):
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        # TODO: predicted
        predicted = (inputs @ w) + b
        actual = y[start:end]
        # TODO: calcualte MSE loss
        loss = loss_functions.MeanSquaredError(actual, predicted)

        # TODO: backward
        # hint you need to just do loss.backward()

        grad = np.ones(loss.shape)
        loss.backward(grad)
        epoch_loss += loss.data.sum()

        # TODO: update w and b (Don't use 'w -= ' and use ' w = w - ...') (you don't need to use optim.SGD in this task)
        w = w - w.grad * Tensor([learning_rate])
        b = b - b.grad * Tensor([learning_rate])

# print(coef)
# print(w)
# print(b)
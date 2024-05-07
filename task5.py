# Task 5
import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

# TODO: define a linear layer using Linear() class  
l = Linear(3,1)


# TODO: define an optimizer using SGD() class 
optimizer = SGD([l],learning_rate=0.1)

# TODO: print weight and bias of linear layer
print(l.weight,end="\n")
print(l.bias,end="\n\n\n")

learning_rate = 0.1
batch_size = 20

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size


        #print(start, end)

        inputs = X[start:end]

        # TODO: predicted
        predicted = l(inputs)
        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)

        # TODO: calcualte MSE loss
        loss = loss_functions.MeanSquaredError(actual, predicted)

        # TODO: backward
        # hint you need to just do loss.backward()
        optimizer.zero_grad()
        grad = np.ones(loss.shape)
        loss.backward(grad)


        # TODO: add loss to epoch_loss
        epoch_loss += loss.data.sum()


        # TODO: update w and b using optimizer.step()
        optimizer.step()
        

# TODO: print weight and bias of linear layer
print(l.weight,end="\n")
print(l.bias,end="\n")

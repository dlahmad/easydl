import numpy as np
import easydl as edl
from easydl.tensor import tensor
from easydl.nodes.multiply import Multiply
from easydl.nodes.add import Add
import timeit
from easydl.optimizer import Optimizer
from easydl.tape import Tape
from easydl.nodes.layers.dense import Dense
from easydl.nodes.activations.sigmoid import Sigmoid
from easydl.optimizers.sgd import Sgd


def test_func():

    a = tensor(0.5 * np.ones((1, 10)))
    for i in range(10000):
        with Tape() as tape:
            d = s(l(a))
            print(d.numpy)
        d.backward()
        optimizer.optimize(tape)


edl.init_easydl()

optimizer = Sgd(learning_rate=0.1)

l = Dense(10, 20)
s = Sigmoid()

test_func()
#print(timeit.Timer(test_func).timeit(100)/100)
#print('grad a: {} grad b: {}'.format(a.grad, b.grad))

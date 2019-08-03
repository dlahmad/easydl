import numpy as np
import easydl as edl
from easydl.tensor import tensor
from easydl.nodes.multiply import Multiply
from easydl.nodes.add import Add
import timeit
from easydl.optimizer import Optimizer
from easydl.tape import Tape
from easydl.nodes.layers import Dense
from easydl.nodes.losses import MSE
from easydl.nodes.activations import Sigmoid, Softmax
from easydl.optimizers.sgd import Sgd
import matplotlib.pyplot as plt
import imageio
from numba import jit


def test_func():
    target = tensor(0.2 * np.ones((1, 10)))
    a = tensor(0.5 * np.ones((1, 10)))
    for i in range(100000):
        with Tape() as tape:
            d = s(l(a))

            r = mse([d, target])
            print(r.numpy)

        r.backward()
        optimizer.optimize(tape)


edl.init_easydl()
optimizer = Sgd(learning_rate=0.001)

l = Dense(10, 10)
l2 = Dense(20, 10)
s = Sigmoid()
mse = MSE()

test_func()
print()
import numpy as np
import easydl as edl
from easydl.tensor import tensor
from easydl.nodes.multiply import Multiply
from easydl.nodes.add import Add
import timeit
from easydl.optimizer import Optimizer


def test_func():
    with Optimizer() as opt:
        a = tensor(2 * np.ones((1, 1)))
        b = tensor(3 * np.ones((1, 1)))
        c = tensor(4 * np.ones((1, 1)))

        d = (a + a*b)**2 * c

    d.backward()


edl.init_easydl()

print(timeit.Timer(test_func).timeit(100)/100)
#print('grad a: {} grad b: {}'.format(a.grad, b.grad))

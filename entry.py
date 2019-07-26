import numpy as np
from easydl.tensor import tensor
from easydl.nodes.multiply import Multiply
from easydl.nodes.add import Add

a = tensor(2 * np.ones((1, 1)))
b = tensor(3 * np.ones((1, 1)))
c = tensor(4 * np.ones((1, 1)))


d = (a + a*b)**2 * c

d.backward()

print('grad a: {} grad b: {}'.format(a.grad, b.grad))

import torch
import torch.nn as nn
import torchvision.datasets as dataset
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
from easydl.nodes.activations import Sigmoid, Softmax, ReLu
from easydl.optimizers.sgd import Sgd
import matplotlib.pyplot as plt


def eval_func(data, label):
    batch_size = 1
    counter = 0.0
    correct = 0.0
    for i in range(0, data.shape[0], batch_size):
        target = tensor(label[i:batch_size + i])
        source = tensor(data[i:batch_size + i])
        d = re(l2(re(l(source))))

        pred = np.argmax(d.numpy)
        true_pred = target.numpy.flatten()
        counter += 1
        correct += (pred == true_pred)

    print(correct/counter)


def test_func(data, label):

    for e in range(30):
        batch_size = 4
        for i in range(0, data.shape[0], batch_size):

            target = tensor(label[i:batch_size+i])
            source = tensor(data[i:batch_size+i])

            with Tape() as tape:
                d = s(l2(re(l(source))))

                # d.to_gpu()
                # target.to_gpu()
                # mse.to_gpu()

                r = mse([d, target])
                print(np.sum(r.numpy))

            r.backward()
            optimizer.optimize(tape)


edl.init_easydl()
optimizer = Sgd(learning_rate=0.6, momentum=0.1)

l = Dense(784, 128)
l.build()
#l.to_gpu()
l2 = Dense(128, 10)
re = ReLu()
re2 = ReLu()
s = Sigmoid()
so = Softmax()
mse = MSE()

data_set = dataset.MNIST('./datasets/mnist', download=True)
data = data_set.train_data.numpy()
data = np.reshape(data, (data.shape[0], -1)) / 255.
label = data_set.train_labels.numpy()
label_raw = np.zeros((label.shape[0], 10))
label_raw[np.arange(label.shape[0]), label] = 1
test_func(data[:5000], label_raw[:5000])
eval_func(data[:1000], label[:1000])


print()
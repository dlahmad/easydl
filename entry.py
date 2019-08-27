import numpy as np
import torchvision.datasets as dataset

import easydl as edl
from easydl.nodes.activations import Sigmoid, Softmax, ReLu
from easydl.nodes.layers import Dense, Conv2d
from easydl.nodes.losses import MSE, CrossEntropySoftmax
from easydl.optimizers import Sgd
from easydl import Tape, tensor


def eval_func(data, label):
    batch_size = 1
    counter = 0.0
    correct = 0.0
    for i in range(0, data.shape[0], batch_size):
        target = tensor(label[i:batch_size + i])
        source = tensor(data[i:batch_size + i])
        d = so(l3(re(l2(re(l(source))))))

        pred = np.argmax(d.numpy)
        true_pred = target.numpy.flatten()
        counter += 1
        correct += (pred == true_pred)

    print(correct/counter)


def test_func(data, label):

    for e in range(20):
        batch_size = 24
        for i in range(0, data.shape[0], batch_size):

            target = tensor(label[i:batch_size+i])
            source = tensor(data[i:batch_size+i])

            with Tape() as tape:
                d = l3(re(l2(re(l(source)))))

                # d.to_gpu()
                # target.to_gpu()
                # mse.to_gpu()

                r = ces([d, target])
                print(np.sum(r.numpy) / batch_size)

            r.backward()
            optimizer.optimize(tape)


edl.init_easydl(False)
optimizer = Sgd(learning_rate=0.4, momentum=0.9)


conv = Conv2d(10, 10, 3, 10, (3, 3), paddings=(1, 1))


conv.build()

z = np.ones((2, 10, 10, 3))
conv.forward([z], 2)

l = Dense(784, 128)
#l.to_gpu()
l2 = Dense(128, 10)
l3 = Dense(10, 10)
re = ReLu()
re2 = ReLu()
s = Sigmoid()
so = Softmax()
mse = MSE()
ces = CrossEntropySoftmax()

data_set = dataset.MNIST('./datasets/mnist', download=True)
data = data_set.data.numpy()
data = np.reshape(data, (data.shape[0], -1)) / 255.
label = data_set.targets.numpy()
label_raw = np.zeros((label.shape[0], 10))
label_raw[np.arange(label.shape[0]), label] = 1
test_func(data[:5000], label_raw[:5000])
eval_func(data[:1000], label[:1000])


print()
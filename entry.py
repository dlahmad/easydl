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
import matplotlib.pyplot as plt
import imageio
from numba import jit

def test_func():

    a = tensor(0.5 * np.ones((1, 10)))
    for i in range(10000):
        with Tape() as tape:
            d = s(l2(l(a)))
            print(d.numpy)

        d.backward()
        optimizer.optimize(tape)


edl.init_easydl()
optimizer = Sgd(learning_rate=0.001)

l = Dense(10, 20)
l2 = Dense(20, 10)
s = Sigmoid()

test_func()
print()
#print(timeit.Timer(test_func).timeit(100)/100)
#print('grad a: {} grad b: {}'.format(a.grad, b.grad))


def im2col_indices(kernel_shape, img_shape, strides=(1, 1), paddings=(0, 0), dilations=(1, 1)):
    f_h, f_w = kernel_shape
    img_c, img_h, img_w = img_shape
    stride_h, stride_w = strides
    padding_h, padding_w = paddings
    dilation_h, dilation_w = dilations

    p_img_h = int(img_h + 2 * padding_h)
    p_img_w = int(img_w + 2 * padding_w)
    index_list = np.arange(0, p_img_h * p_img_w)
    index_image = np.reshape(index_list, (p_img_h, p_img_w))

    h_offset = int(f_h / 2)
    w_offset = int(f_w / 2)

    img_f_h = int((p_img_h - 2 * h_offset) / stride_h)
    img_f_w = int((p_img_w - 2 * w_offset) / stride_w)

    indices = np.zeros((img_f_h, img_f_w, f_h, f_w), dtype=int)
    print(indices)

    for h in range(h_offset, p_img_h - h_offset, stride_h):
        for w in range(w_offset, p_img_w - w_offset, stride_w):
            for h_f in range(-h_offset, -h_offset + f_h, dilation_h):
                for w_f in range(-w_offset, -w_offset + f_w, dilation_w):
                    indices[int(h / stride_h) - h_offset, int(w / stride_w) - w_offset, h_f + h_offset, w_f + w_offset] \
                        = index_image[h + h_f, w + w_f]

    c_indices = np.tile(indices, (img_c, 1, 1, 1, 1))
    offset = np.reshape(np.arange(img_c) * img_h * img_w, (img_c, 1, 1, 1, 1))
    c_indices += offset

    return (img_c, img_f_h, img_f_w, f_h, f_w), c_indices


def im2col(img: np.ndarray, kernel_shape, img_shape, strides=(1, 1), paddings=(0, 0), dilations=(1, 1)):
    new_img_shape, indices = im2col_indices(kernel_shape, img_shape, strides, paddings, dilations)

    img_flat = img.flatten()
    ind_flat = indices.flatten()

    sobel = np.array([[1, 0, -1, 2, 0, -2, 1, 0, -1]]).T

    mat = np.reshape(img_flat[ind_flat], (-1, new_img_shape[-1] * new_img_shape[-2]))

    res = np.dot(mat, sobel)

    new_img = np.reshape(res, new_img_shape[:3])

    img_c = np.transpose(img, (1, 2, 0))
    img_c /= np.max(np.abs(img_c))
    n_c = np.transpose(new_img, [1, 2, 0])
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_c)
    ax[1].imshow(n_c)

    fig.show()
    print()

    print(mat.shape)


img = imageio.imread('/home/vedaevolution/Downloads/test.png')

img = np.transpose(img, (2, 0, 1)) / 255

img_c = img.shape[0]
img_h = img.shape[1]
img_w = img.shape[2]
img_shape = (img_c, img_h, img_w)
kernel_h = 3
kernel_w = 3
kernel_shape = (kernel_h, kernel_w)

# img = np.zeros(img_shape)
#
# img[1, 1:2, 1:2] = 1
# img[1, 3:4, 3:4] = 1

# new_img_shape, indices = im2col_indices(kernel_shape, img_shape, paddings=(0, 0), strides=(2, 2))
#
# ind_map = np.reshape(indices, new_img_shape[:3] + (9,))

pad_img = img  # np.pad(img, (0, 0), 'constant', constant_values=(0, 0))

im2col(pad_img, kernel_shape, img_shape, paddings=(0, 0), strides=(1, 1))

# pad_img_flat = pad_img.flatten()
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img[1])
#
# new_img = np.zeros((ind_map.shape[0], ind_map.shape[1], ind_map.shape[2]))
#
# for c in range(ind_map.shape[0]):
#     for h in range(ind_map.shape[1]):
#         for w in range(ind_map.shape[2]):
#             new_img[c, h, w] = pad_img_flat[ind_map[c, h, w]][4]
#
# ax[1].imshow(new_img[1])
# fig.show()
# print()
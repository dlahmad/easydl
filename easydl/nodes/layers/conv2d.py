from typing import Union, List
from ..layer import Layer
from ...initializer import Initializer
from ...initializers.random_uniform import RandomUniform
import numpy as np


class Conv2d(Layer):

    def __init__(self):
        super().__init__()

    @staticmethod
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
                        indices[
                            int(h / stride_h) - h_offset, int(w / stride_w) - w_offset, h_f + h_offset, w_f + w_offset] \
                            = index_image[h + h_f, w + w_f]

        c_indices = np.tile(indices, (img_c, 1, 1, 1, 1))
        offset = np.reshape(np.arange(img_c) * img_h * img_w, (img_c, 1, 1, 1, 1))
        c_indices += offset

        return (img_c, img_f_h, img_f_w, f_h, f_w), c_indices

    @staticmethod
    def im2col(img: np.ndarray, kernel_shape, img_shape, strides=(1, 1), paddings=(0, 0), dilations=(1, 1)):
        new_img_shape, indices = Conv2d.im2col_indices(kernel_shape, img_shape, strides, paddings, dilations)

        img_flat = img.flatten()
        ind_flat = indices.flatten()

        mat = np.reshape(img_flat[ind_flat], (-1, new_img_shape[-1] * new_img_shape[-2]))

        return mat

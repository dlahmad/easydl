from typing import List, Union, Tuple

import numpy as np

from ..layer import Layer


class Conv2d(Layer):

    def __init__(self, width, height, input_channels, output_channels,
                 kernel_shape, strides=(1, 1), paddings=(0, 0), dilations=(1, 1)):
        super().__init__()
        self.width = width
        self.height = height
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.img_shape = (self.input_channels, self.height, self.width)
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.paddings = paddings
        self.dilations = dilations
        self.im2col_indices = None
        self.new_shape = None

    def build(self) -> None:
        (self.new_shape, self.im2col_indices, ) = \
            self.compute_im2col_indices(self.img_shape, self.kernel_shape,
                                        self.strides, self.paddings, self.dilations)

        self.variables['w'] = \
            self.init_variable(self.output_channels, self.input_channels
                               , self.kernel_shape[0], self.kernel_shape[1])

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int) -> \
            Tuple[np.ndarray, Union[None, np.ndarray, List[np.ndarray]]]:

        images = inputs[0]
        filters = self.variables['w'].reshape((self.output_channels, -1)).T
        im_col = self.im2col(images, batch_size).dot(filters)
        new_images = im_col.reshape((batch_size, self.new_shape[0], self.new_shape[1], self.output_channels))

        return new_images

    def compute_im2col_batch_indices(self, batch_size):
        b_indices = np.tile(self.im2col_indices, (batch_size, 1, 1, 1, 1, 1))
        offset = np.reshape(np.arange(batch_size) * self.width * self.height * self.input_channels,
                            (batch_size, 1, 1, 1, 1, 1))
        b_indices += offset
        return b_indices

    @staticmethod
    def compute_im2col_indices(img_shape, kernel_shape, strides=(1, 1),
                               paddings=(0, 0), dilations=(1, 1)):
        f_h, f_w = kernel_shape
        img_c, img_h, img_w = img_shape
        stride_h, stride_w = strides
        padding_h, padding_w = paddings
        dilation_h, dilation_w = dilations

        p_img_h = int(img_h + 2 * padding_h)
        p_img_w = int(img_w + 2 * padding_w)
        index_list = np.arange(0, p_img_h * p_img_w * img_c)
        index_image = np.reshape(index_list, (p_img_h, p_img_w, img_c))

        h_offset = int(f_h / 2)
        w_offset = int(f_w / 2)

        img_f_h = int((p_img_h - 2 * h_offset) / stride_h)
        img_f_w = int((p_img_w - 2 * w_offset) / stride_w)

        indices = np.zeros((img_f_h, img_f_w, img_c, f_h, f_w), dtype=int)

        for h in range(h_offset, p_img_h - h_offset, stride_h):
            for w in range(w_offset, p_img_w - w_offset, stride_w):
                for c in range(img_c):
                    for h_f in range(-h_offset, -h_offset + f_h, dilation_h):
                        for w_f in range(-w_offset, -w_offset + f_w, dilation_w):
                            indices[
                                int(h / stride_h) - h_offset, int(w / stride_w) - w_offset,
                                c, h_f + h_offset, w_f + w_offset] \
                                = index_image[h + h_f, w + w_f, c]

        return (img_f_h, img_f_w, img_c, f_h, f_w), indices#c_indices

    def im2col(self, img: np.ndarray, batch_size):
        padded_img = self.np.pad(img, ((0, 0), self.paddings, self.paddings, (0, 0)),
                                 'constant')
        img_flat = padded_img.flatten()
        ind_flat = self.compute_im2col_batch_indices(batch_size).flatten()

        mat = np.reshape(img_flat[ind_flat], (-1, self.new_shape[-1] * self.new_shape[-2] * self.new_shape[-3]))

        return mat

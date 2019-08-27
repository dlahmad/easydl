from typing import Union, Sequence, Tuple

import numpy as np

from ..activation import Activation
from ...util.input_check import check_arg_number


class Softmax(Activation):

    def __init__(self):
        super().__init__()

    def input_check(self, inputs: Union[np.ndarray, Sequence[np.ndarray]]) -> None:
        check_arg_number(inputs, 1)

    def forward(self, inputs: Union[np.ndarray, Sequence[np.ndarray]], batch_size: int) -> \
            Tuple[np.ndarray, Union[None, np.ndarray, Sequence[np.ndarray]]]:
        # inputs x

        inp = inputs[0]
        input_size = inp.shape[1]
        v_max = self.np.max(inp)
        add_ones = self.np.ones((input_size, 1))

        exp = self.np.exp(inp - v_max)  # f

        exp_sum = (self.np.dot(exp, add_ones) + self.eps).reshape((-1, 1))  # g and h=exp

        sm = exp / exp_sum  # y

        return sm, (exp, exp_sum, add_ones)

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, Sequence[np.ndarray]], batch_size)\
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        exp, exp_sum, add_ones = cache

        d_y_d_x = self.np.multiply(gradients, exp) / exp_sum

        d_y_d_g = self.np.dot(self.np.multiply(exp, gradients), -add_ones) / self.np.power(exp_sum, 2)
        d_y_d_h = self.np.dot(d_y_d_g, add_ones.T)
        d_h_d_x = self.np.multiply(d_y_d_h, exp)

        d_y_d_x = d_y_d_x + d_h_d_x

        return d_y_d_x

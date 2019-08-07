from typing import Union, List
from ..activation import Activation
import numpy as np
from ...util.input_check import check_arg_number
from ...util.numerics import get_eps


class Softmax(Activation):

    def __init__(self):
        super().__init__()
        self.eps = get_eps()

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 1)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        # inputs x

        inp = inputs[0]
        input_size = inp.shape[1]
        v_max = self.np.max(inp)
        add_ones = self.np.ones((input_size, 1))

        exp = self.np.exp(inp - v_max)  # f

        exp_sum = (np.dot(exp, add_ones) + self.eps).reshape((-1, 1))  # g and h=exp

        sm = exp / exp_sum  # y

        return sm, (exp, exp_sum, add_ones)

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        exp, exp_sum, add_ones = cache

        d_y_d_x = self.np.multiply(gradients, exp) / exp_sum

        d_y_d_g = self.np.dot(self.np.multiply(exp, gradients), -add_ones) / self.np.power(exp_sum, 2)
        d_y_d_h = self.np.dot(d_y_d_g, add_ones.T)
        d_h_d_x = self.np.multiply(d_y_d_h, exp)

        d_y_d_x = d_y_d_x + d_h_d_x

        return d_y_d_x

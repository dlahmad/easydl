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

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]):

        # inputs x
        exp = self.np.exp(inputs)  # f

        exp_sum = (self.np.sum(exp, 1) + self.eps).reshape((-1, 1))  # g and h=exp

        sm = exp / exp_sum  # y

        return sm, (exp, exp_sum, sm)

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]]):
        exp, exp_sum, sm = cache

        d_y_d_f = gradients / (exp + self.eps)
        d_y_d_x = d_y_d_f

        d_y_d_g = np.dot(exp + gradients, -np.ones((sm.shape[1], 1))) / (np.power(exp_sum, 2) + self.eps)
        d_y_d_h = np.dot(d_y_d_g, np.ones((1, exp.shape[1])))
        d_h_d_x = d_y_d_h

        d_y_d_x = d_y_d_x + d_h_d_x

        return d_y_d_x

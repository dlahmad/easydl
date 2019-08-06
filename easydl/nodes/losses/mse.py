from ..loss import Loss
from ...util.input_check import check_arg_number, check_equal_shape
from typing import Union, List
import numpy as np


class MSE(Loss):

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 2)
        check_equal_shape(inputs)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        inp_0, inp_1 = inputs

        difference = inp_0 - inp_1
        squared = np.power(difference, 2)
        res = np.mean(squared, axis=tuple(range(1, squared.ndim)))

        return res, difference

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        difference = cache
        size = difference.size
        expand_shape = (1, ) * (difference.ndim-1)
        gradients = gradients.reshape((-1,) + expand_shape)

        scaled_difference = 2*gradients * difference

        return scaled_difference, -scaled_difference


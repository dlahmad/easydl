from ..loss import Loss
from ...util.input_check import check_arg_number, check_equal_shape
from typing import Union, List
import numpy as np


class MSE(Loss):

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 2)
        check_equal_shape(inputs)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]):
        inp_0, inp_1 = inputs

        difference = inp_0 - inp_1
        squared = np.power(difference, 2)
        res = np.mean(squared, axis=tuple(range(1, squared.ndim)))

        return res, difference

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]]):
        difference = cache
        size = difference.size

        scaled_difference = 2*gradients/size * difference

        return scaled_difference, -scaled_difference


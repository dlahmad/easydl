from typing import Union, List

import numpy as np

from ..activation import Activation
from ...util.input_check import check_arg_number


class Sigmoid(Activation):

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 1)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        res = 1. / (1 + self.np.exp(-inputs[0]))

        return res, res

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        res = cache
        return np.multiply(gradients, np.multiply(res, (1 - res)))


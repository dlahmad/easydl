from typing import Union, List
from ..activation import Activation
import numpy as np
from ...util.input_check import check_arg_number


class Softmax(Activation):

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 1)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]):
        exp = 1./(1 + self.np.exp(-inputs))

        return res, res

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]]):
        res = cache
        return res * (1 - res)
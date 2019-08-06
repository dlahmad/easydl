from typing import Union, List
from ..activation import Activation
import numpy as np
from ...util.input_check import check_arg_number


class ReLu(Activation):

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 1)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        mask = (inputs[0] >= 0)
        res = inputs[0] * mask

        return res, mask

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        mask = cache
        return mask * gradients

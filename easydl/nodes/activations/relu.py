from typing import Union, List, Tuple

import numpy as np

from ..activation import Activation
from ...util.input_check import check_arg_number


class ReLu(Activation):

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 1)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int) -> \
            Tuple[np.ndarray, Union[None, np.ndarray, List[np.ndarray]]]:
        mask = (inputs[0] >= 0)
        res = inputs[0] * mask

        return res, mask

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size)\
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        mask = cache
        return mask * gradients

from typing import Union, Sequence, Tuple

import numpy as np

from ...node import Node
from ...util.input_check import check_arg_number


class Power(Node):

    def __init__(self, power: float):
        super().__init__()
        self.power: float = power

    def build(self):
        pass

    def input_check(self, inputs: np.ndarray) -> None:
        check_arg_number(inputs, 1, self)

    def forward(self, inputs: Union[np.ndarray, Sequence[np.ndarray]], batch_size: int) -> \
            Tuple[np.ndarray, Union[None, np.ndarray, Sequence[np.ndarray]]]:
        return self.np.power(inputs[0], self.power)[0], inputs[0]

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, Sequence[np.ndarray]], batch_size)\
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return gradients * self.power * cache


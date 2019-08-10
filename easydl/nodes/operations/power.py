from typing import Union, List

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

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        return self.np.power(inputs[0], self.power)[0], inputs[0]

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        return gradients * self.power * cache


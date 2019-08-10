from typing import Union, List

import numpy as np

from ...node import Node
from ...util.input_check import check_arg_number


class Substract(Node):

    def __init__(self):
        super().__init__()

    def build(self):
        pass

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 2, self)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        return self.np.add(inputs[0], -inputs[1]), None

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        return gradients, -gradients


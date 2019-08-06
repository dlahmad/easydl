from ..node import Node
from typing import Union, List
from ..util.input_check import check_arg_number, check_equal_shape
import numpy as np
from ..util.numerics import get_eps


class Divide(Node):

    def __init__(self):
        super().__init__()

    def build(self):
        pass

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 2, self)
        check_equal_shape(inputs, self)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        return self.np.multiply(inputs[0], inputs[1]), inputs

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        return gradients * cache[1], (gradients * -cache[0])/(np.power(cache[1], 2) + get_eps())


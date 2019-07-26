from ..node import Node
from typing import Union, List
from ..util.input_check import check_arg_number, check_equal_shape
import numpy as np


class Multiply(Node):

    def __init__(self):
        super().__init__()

    def build(self):
        pass

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 2, self)
        check_equal_shape(inputs, self)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]):
        return self.np.multiply(inputs[0], inputs[1])

    def backward(self, gradients: np.ndarray, inputs: Union[np.ndarray, List[np.ndarray]]):
        return gradients * inputs[1], gradients * inputs[0]

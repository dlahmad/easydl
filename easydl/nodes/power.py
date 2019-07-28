from ..node import Node
from typing import Union, List
from ..util.input_check import check_arg_number, check_arg_shape
import numpy as np


class Power(Node):

    def __init__(self, power: float):
        super().__init__()
        self.power: float = power

    def build(self):
        pass

    def input_check(self, inputs: np.ndarray) -> None:
        check_arg_number(inputs, 1, self)

    def forward(self, inputs: np.ndarray):
        return self.np.power(inputs, self.power)[0], inputs

    def backward(self, gradients: np.ndarray, cache: np.ndarray):
        return gradients * self.power * cache


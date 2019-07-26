from ..node import Node
from typing import Union, List
from ..util.input_check import check_arg_number, check_arg_shape
import numpy as np


class Add(Node):

    def __init__(self):
        super().__init__()

    def build(self):
        pass

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 2, self)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]):
        return self.np.add(inputs[0], inputs[1])

    def backward(self, gradients: np.ndarray, inputs: Union[np.ndarray, List[np.ndarray]]):
        return gradients, gradients

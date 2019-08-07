from ..loss import Loss
from ...util.input_check import check_arg_number, check_equal_shape
from typing import Union, List
import numpy as np


class CrossEntropySoftmax(Loss):

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        check_arg_number(inputs, 2)
        check_equal_shape(inputs)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int):
        pass

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size):
        pass
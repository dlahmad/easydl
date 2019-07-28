from easydl.node import Node
from typing import Union, List
from easydl.util.input_check import check_arg_number, check_arg_shape
import numpy as np
from easydl.abstract_tensor import AbstractTensor


class Constant(AbstractTensor):
    def __init__(self, array: np.ndarray):
        super().__init__(array, False)

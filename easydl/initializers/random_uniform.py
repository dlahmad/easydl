from ..initializer import Initializer
import numpy as np
from typing import Union, Tuple, List


class RandomUniform(Initializer):

    def init_variable(self, size: List[int]) -> np.ndarray:
        return np.random.rand(*size)

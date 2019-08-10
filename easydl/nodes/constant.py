import numpy as np

from ..abstract_tensor import AbstractTensor


class Constant(AbstractTensor):
    def __init__(self, array: np.ndarray):
        super().__init__(array, None)

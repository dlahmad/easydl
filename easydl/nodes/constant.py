import numpy as np
from easydl.abstract_tensor import AbstractTensor


class Constant(AbstractTensor):
    def __init__(self, array: np.ndarray):
        super().__init__(array, None)

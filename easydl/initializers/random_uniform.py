from typing import List

import numpy as np

from ..initializer import Initializer


class RandomUniform(Initializer):

    def init_variable(self, size: List[int]) -> np.ndarray:
        return self.np.random.uniform(0, 1, size)

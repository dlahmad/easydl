from typing import Sequence

import numpy as np

from ..initializer import Initializer


class RandomUniform(Initializer):

    def init_variable(self, size: Sequence[int]) -> np.ndarray:
        return self.np.random.uniform(0, 1, size)

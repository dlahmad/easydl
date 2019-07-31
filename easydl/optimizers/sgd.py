from typing import List, Set, Union, Tuple
from ..optimizer import Optimizer
import numpy as np


class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate: float = learning_rate

    def step(self, variable: np.ndarray, gradient: np.ndarray, state: Union[None, np.ndarray]) ->\
            Tuple[np.ndarray, np.ndarray]:
        return variable - self.learning_rate * gradient, None


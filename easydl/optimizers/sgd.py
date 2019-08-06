from typing import List, Set, Union, Tuple
from ..optimizer import Optimizer
import numpy as np


class Sgd(Optimizer):

    def __init__(self, learning_rate, momentum=0.9):
        super().__init__()
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum

    def step(self, variable: np.ndarray, gradient: np.ndarray, state: Union[None, np.ndarray]) ->\
            Tuple[np.ndarray, Union[None, np.ndarray]]:
        if state is None:
            state = gradient

        moving_momentum = (1 - self.momentum) * gradient + state * self.momentum

        return variable - self.learning_rate * moving_momentum, moving_momentum


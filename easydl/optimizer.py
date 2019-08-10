from typing import Tuple, Union

import numpy as np

from .abstract_object import AbstractObject
from .tape import Tape


class Optimizer(AbstractObject):

    def optimize(self, tape: Tape):
        for node, instance in tape.gradient_operations:
            for key in node.variables.keys():
                var = node.variables[key]
                grad = node.gradients[key]
                state = node.optimizer_cache[key] if key in node.optimizer_cache else None

                node.variables[key], node.optimizer_cache[key] = self.step(var, grad, state)

                node.gradients[key].fill(0)

    def step(self, variable: np.ndarray, gradient: np.ndarray, state: Union[None, np.ndarray])\
            -> Tuple[np.ndarray, np.ndarray]:
        pass


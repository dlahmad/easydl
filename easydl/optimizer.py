from typing import List, Set, Tuple, Union
from .tape import Tape
from .node import Node
import numpy as np
from .abstract_object import AbstractObject


class Optimizer(AbstractObject):

    def optimize(self, tape: Tape):
        for node in tape.operations:
            for key in node.variables.keys():
                var = node.variables[key]
                grad = node.gradients[key]
                state = node.optimizer_cache[key] if key in node.optimizer_cache else None

                node.variables[key], node.optimizer_cache[key] = self.step(var, grad, state)

    def step(self, variable: np.ndarray, gradient: np.ndarray, state: Union[None, np.ndarray])\
            -> Tuple[np.ndarray, np.ndarray]:
        pass


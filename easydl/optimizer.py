from typing import Tuple, Union

import numpy as np

from .abstract_object import AbstractObject
from .tape import Tape


class Optimizer(AbstractObject):
    """
    Optimizer class for optimizing node variables. Every optimizer
    should derive this class.
    """

    def optimize(self, tape: Tape) -> None:
        """
        Optimizes the variables of the nodes recorded by the tape provided.
        :param tape: Tape that recorded the nodes used.
        """
        for node, instance in tape.gradient_operations:
            for key in node.variables.keys():
                var = node.variables[key]
                grad = node.gradients[key]
                state = node.optimizer_cache[key] if key in node.optimizer_cache else None

                node.variables[key], node.optimizer_cache[key] = self.step(var, grad, state)

                node.gradients[key].fill(0)

    def step(self, variable: np.ndarray, gradient: np.ndarray, state: Union[None, np.ndarray])\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes one optimization steps using the gradients provided by the tape. This
        method is to be implemented for a new optimizer.
        :param variable: Variable to optimize.
        :param gradient: Gradient for the variable to optimize.
        :param state: State of the optimizer for the variable.
        :return: Optimized variable and new state.
        If the optimizer doesn't use a state it returns 'none'.

        """
        pass


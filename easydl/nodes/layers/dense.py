from typing import Union, List, Tuple

import numpy as np

from ..layer import Layer
from ...initializer import Initializer


class Dense(Layer):

    def __init__(self, input_dim: int, output_dim: int, initializer: Union[None, Initializer] = None):
        super().__init__(initializer=initializer)
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

    def build(self) -> None:
        self.variables['w'] = self.init_variable(self.input_dim, self.output_dim)
        self.variables['b'] = self.init_variable(self.output_dim)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]], batch_size: int) -> \
            Tuple[np.ndarray, Union[None, np.ndarray, List[np.ndarray]]]:
        return self.np.dot(inputs[0], self.variables['w']) + self.variables['b'], inputs[0]

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]], batch_size)\
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        inp = cache
        propagating_gradient = self.np.dot(gradients, self.variables['w'].T)
        self.gradients['w'] = self.np.dot(inp.T, gradients) / batch_size
        self.gradients['b'] = np.sum(gradients, axis=0) / batch_size

        return propagating_gradient


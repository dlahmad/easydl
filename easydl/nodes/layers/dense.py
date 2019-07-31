from typing import Union, List
from ..layer import Layer
from ...initializer import Initializer
from ...initializers.random_uniform import RandomUniform
import numpy as np


class Dense(Layer):

    def __init__(self, input_dim: int, output_dim: int, initializer: Union[None, Initializer] = None):
        super().__init__(initializer=initializer)
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

    def build(self) -> None:
        self.variables['w'] = self.init_variable(self.input_dim, self.output_dim)
        self.variables['b'] = self.init_variable(self.output_dim)

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]):
        return self.np.dot(inputs, self.variables['w']) + self.variables['b'], inputs

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, List[np.ndarray]]):
        inp = cache
        propagating_gradient = self.np.dot(gradients, self.variables['w'].T)
        self.gradients['w'] = self.np.dot(inp.T, gradients)
        self.gradients['b'] = gradients

        return propagating_gradient


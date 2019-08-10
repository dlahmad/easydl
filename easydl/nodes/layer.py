from typing import Union

import numpy as np

from ..initializer import Initializer
from ..initializers.random_uniform import RandomUniform
from ..node import Node


class Layer(Node):

    def __init__(self, initializer: Union[None, Initializer] = None):
        self.initializer: Initializer = initializer if initializer else RandomUniform()
        super().__init__()
        self.needs_gradient: bool = True
        self.needs_init: bool = True

    def init_variable(self, *shape: int) -> np.ndarray:
        init = self.initializer.init_variable(list(shape))
        init /= init.size
        return init




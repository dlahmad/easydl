from ..node import Node
from ..tape import Tape
from ..tensor import AbstractTensor
from ..initializer import Initializer
from ..initializers.random_uniform import RandomUniform
from typing import Union, List, Tuple
import numpy as np


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

    def __call__(self, args: Union[AbstractTensor, List[AbstractTensor]]):
        if self.needs_gradient:
            Tape.add_node(self)
        return super().__call__(args)


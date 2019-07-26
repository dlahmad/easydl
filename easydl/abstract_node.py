import numpy as np
from typing import Union, List


class AbstractNode:

    def __init__(self):
        self.np: np = np
        self.name: Union[str, None] = None
        self.needs_gradient: bool = True
        self.name = self.__class__.__name__
        self.built = False

    def input_check(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> None:
        pass

    def build(self) -> None:
        pass

    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]):
        pass

    def backward(self, gradients: np.ndarray, inputs: Union[np.ndarray, List[np.ndarray]]):
        pass

    def _internal_backward(self, output_tensor, gradients: np.ndarray):
        pass






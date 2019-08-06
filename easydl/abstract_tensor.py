from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple
from .abstract_node import AbstractNode
from .abstract_object import AbstractObject


class AbstractTensor(AbstractObject):

    def __init__(self, arr: np.ndarray):
        super().__init__()
        self.numpy: np.ndarray = arr
        self.shape: Tuple[int] = self.numpy.shape
        self.origin: Union[AbstractNode, None] = None

    def register_origin(self, node: AbstractNode):
        self.origin = node

    def __call__(self, *args, **kwargs):
        pass

    def backward(self, gradient: Union[np.ndarray, List[np.ndarray]] = None):

        if self.origin is None:
            raise Exception('You can not back propagate a tensor with no origin!')

        if gradient is None:
            self.origin.raw_backward(self, np.ones(self.numpy.shape))
        elif gradient.shape == self.numpy.shape:
            modified_grad = self.numpy * gradient
            self.origin.raw_backward(self, modified_grad)
        else:
            raise Exception('Tensor created from node {} got a gradient with an invalid shape!'
                            .format(self.origin.name))
import numpy as np
import cupy as cp
from typing import Union, List, Tuple
from .abstract_node import AbstractNode
from .abstract_object import AbstractObject


class AbstractTensor(AbstractObject):

    def __init__(self, arr: np.ndarray, use_gpu=None):
        super().__init__()
        if use_gpu is not None:
            self._use_gpu = use_gpu
            self.np = cp if use_gpu else np

        self.numpy: np.ndarray = self.convert_to_device(arr)
        self.shape: Tuple[int] = self.numpy.shape
        self.origin: Union[AbstractNode, None] = None

    def convert_to_device(self, arr):
        arr = self.np.asarray(arr)
        return arr

    def register_origin(self, node: AbstractNode):
        self.origin = node

    def __call__(self, *args, **kwargs):
        pass

    def backward(self, gradient: Union[np.ndarray, List[np.ndarray]] = None):

        if self.origin is None:
            raise Exception('You can not back propagate a tensor with no origin!')

        if gradient is None:
            self.origin.raw_backward(self, self.np.ones(self.numpy.shape))
        elif gradient.shape == self.numpy.shape:
            modified_grad = self.numpy * gradient
            self.origin.raw_backward(self, modified_grad)
        else:
            raise Exception('Tensor created from node {} got a gradient with an invalid shape!'
                            .format(self.origin.name))
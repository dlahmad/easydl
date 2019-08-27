from typing import Union, Sequence, Tuple

import cupy as cp
import numpy as np

from .abstract_node import AbstractNode
from .abstract_object import AbstractObject


class AbstractTensor(AbstractObject):
    """
    Base class for the tensor class. It allows for setting a tensor
    to cpu or gpu and defines some general properties.
    """

    def __init__(self, arr: np.ndarray, use_gpu=None):
        """
        Creates a tensor.
        :param arr: Array to convert to a tensor.
        :param use_gpu: Indicates whether to use cpu or gpu
        arrays.
        """
        super().__init__()
        if use_gpu is not None:
            self._use_gpu = use_gpu
            self.np = cp if use_gpu else np

        self.numpy: np.ndarray = self.convert_to_device(arr)
        """
        Returns the internal array of the tensor. This can be a numpy
        or cupy array.
        """

        self.shape: Tuple[int] = self.numpy.shape
        """
        Shape of the tensor.
        """

        self.origin: Union[AbstractNode, None] = None
        """
        Node which created the tensor.
        """

    def convert_to_device(self, arr) -> np.ndarray:
        """
        Converts array to gpu array if tensor is a gpu tensor.
        :param arr: Array o convert.
        :return: Converted tensor.
        """
        arr = self.np.asarray(arr)
        return arr

    def register_origin(self, node: AbstractNode) -> None:
        """
        Registers a node as the creating node for the current
        tensor.
        :param node: Node the make the creator of this node.
        """
        self.origin = node

    def __call__(self, *args, **kwargs):
        pass

    def backward(self, gradient: Union[np.ndarray, Sequence[np.ndarray]] = None):
        """
        Computes the gradients of all operations which led to the creation
        of the current tensor and where recorded by a tape. This method is
        resulting in an error if there was no tape recording gradients.
        :param gradient:
        """
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


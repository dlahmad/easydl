from typing import Union

import numpy as np

from .abstract_tensor import AbstractTensor
from .nodes.constant import Constant
from .nodes.operations import Add, Substract, Multiply, Divide, Power
from .util.input_check import check_equal_shape


class tensor(AbstractTensor):
    """
    The tensor object ist used for computations. It wraps a numpy or cupy
    array for node operations. If something is to be fed to a node make
    it a tensor first.
    """

    def __init__(self, arr: np.ndarray, use_gpu: bool = None):
        """
        Create a tensor object from a numpy array.
        :type arr: Numpy array to convert to a tensor.
        :type use_gpu: Indicates whether to create a numpy or cupy tensor.
        """
        super().__init__(arr, use_gpu)

    def base_op_two_args(self, op, other: Union[float, AbstractTensor], tmp_first: bool = False):
        """
        Creates a tensor from a constant if needed. It then applies the operation
        specified and returns the resulting tensor.
        :param op: Operation to execute.
        :param other: Other input which either is a constant or a tensor of the same shape.
        :param tmp_first: Indicates if 'other' was provided from the left or right side.
        :return: Tensor resulting from the operation.
        """
        if isinstance(other, (float, int)):
            constant = self.np.ones_like(self.numpy) * other
            tmp_tensor = Constant(constant)
        else:
            check_equal_shape([other.numpy, self.numpy])
            tmp_tensor = other

        args = [tmp_tensor, self] if tmp_first else [self, tmp_tensor]
        return op()(args)

    def __add__(self, other: Union[float, AbstractTensor]):
        return self.base_op_two_args(Add, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union[float, AbstractTensor]):
        return self.base_op_two_args(Substract, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other: Union[float, AbstractTensor]):
        return self.base_op_two_args(Multiply, other)

    def __rmul__(self, other):
        return self.__rmul__(other)

    def __truediv__(self, other: Union[float, AbstractTensor]):
        return self.base_op_two_args(Divide, other)

    def __rtruediv__(self, other):
        return self.base_op_two_args(Divide, other, True)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise Exception('The pow operator for tensors is not defined with the module argument!')
        return Power(power)([self])

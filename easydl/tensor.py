from __future__ import annotations
import numpy as np
from typing import Union
from .abstract_tensor import AbstractTensor
from .nodes.operations import Add, Substract, Multiply, Divide, Power
from .nodes.constant import Constant
from .util.input_check import check_equal_shape


class tensor(AbstractTensor):

    def __init__(self, arr: np.ndarray, use_gpu=None):
        super().__init__(arr, use_gpu)

    def base_op_two_args(self, op, other: Union[float, tensor], tmp_first: bool = False):
        if isinstance(other, (float, int)):
            constant = self.np.ones_like(self.numpy) * other
            tmp_tensor = Constant(constant)
        else:
            check_equal_shape([other.numpy, self.numpy])
            tmp_tensor = other

        args = [tmp_tensor, self] if tmp_first else [self, tmp_tensor]
        return op()(args)

    def __add__(self, other: Union[float, tensor]):
        return self.base_op_two_args(Add, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union[float, tensor]):
        return self.base_op_two_args(Substract, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other: Union[float, tensor]):
        return self.base_op_two_args(Multiply, other)

    def __rmul__(self, other):
        return self.__rmul__(other)

    def __truediv__(self, other: Union[float, tensor]):
        return self.base_op_two_args(Divide, other)

    def __rtruediv__(self, other):
        return self.base_op_two_args(Divide, other, True)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise Exception('The pow operator for tensors is not defined with the module argument!')
        return Power(power)([self])

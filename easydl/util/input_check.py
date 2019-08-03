from typing import List, Union, Tuple
import numpy as np
from ..node import Node
import collections
from .array import can_broadcast
import functools


def check_arg_number(inputs: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
                     arg_number: int, origin: Node = None):
    if isinstance(inputs, (list, tuple)) and not len(inputs) == arg_number\
            or not isinstance(inputs, (list, tuple)) and not arg_number == 1:
        if origin:
            message = 'The node "{}" needs exactly {} inputs!'.format(origin.name, arg_number)
        else:
            message = 'The node needs exactly {} inputs!'.format(arg_number)

        raise Exception(message)


def check_arg_shape(inp: np.ndarray, shape: Tuple[int], origin: Node = None):
    if not inp.shape == shape:
        if origin:
            message = 'The node "{}" expected tensor with shape {} but got shape {}!'\
                .format(origin.name, shape, inp.shape)
        else:
            message = 'The node expected tensor with shape {} but got shape {}!' \
                .format(shape, inp.shape)

        raise Exception(message)


def check_equal_shape(inputs: List[np.ndarray], origin: Node = None):
    reference_shape = inputs[0].shape

    for inp in inputs:
        if not reference_shape == inp.shape:
            if origin:
                message = 'The node "{}" needs inputs with the same shape. At least one input had another shape!' \
                    .format(origin.name)
            else:
                message = 'The node needs inputs with the same shape. At least one input had another shape!'

            raise Exception(message)


def check_can_broad_cast(inp_0: np.ndarray, inp_1: np.ndarray):
    if not can_broadcast(inp_0, inp_1):
        raise Exception('Cannot broadcast shape {} to shape {}!'
                        .format(inp_0.shape, inp_1.shape))


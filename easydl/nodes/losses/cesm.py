from typing import Union, Sequence, Tuple

import numpy as np

from ..loss import Loss
from ...util.input_check import check_arg_number, check_equal_shape


class CrossEntropySoftmax(Loss):

    def input_check(self, inputs: Union[np.ndarray, Sequence[np.ndarray]]) -> None:
        check_arg_number(inputs, 2)
        check_equal_shape(inputs)

    def forward(self, inputs: Union[np.ndarray, Sequence[np.ndarray]], batch_size: int) -> \
            Tuple[np.ndarray, Union[None, np.ndarray, Sequence[np.ndarray]]]:
        inp, label = inputs
        input_size = inp.shape[1]
        v_max = self.np.max(inp)
        add_ones = self.np.ones((input_size, 1))

        exp = self.np.exp(inp - v_max)  # f

        exp_sum = (self.np.dot(exp, add_ones) + self.eps).reshape((-1, 1))  # g and h=exp

        sm = exp / exp_sum  # y

        ce = -np.sum(self.np.log(self.np.maximum(sm, 0.01)) * label, axis=1)[..., self.np.newaxis]

        if ce.max() > 100:
            print('')

        return ce, (label, sm, exp, exp_sum, add_ones)

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, Sequence[np.ndarray]], batch_size)\
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        label, sm, exp, exp_sum, add_ones = cache

        grads = gradients * (sm - label)

        return grads

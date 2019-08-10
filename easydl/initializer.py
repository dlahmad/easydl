from typing import List

import numpy as np

from .abstract_object import AbstractObject


class Initializer(AbstractObject):
    """
    This is the base class for initializer for node variables.
    """

    def init_variable(self, size: List[int]) -> np.ndarray:
        """
        This method is to be overridden by the concrete initializer implementation.
        It creates a numpy or cupy array with the specified size.
        :param size: Size of the array to create.
        :return: Created numpy or cupy array.
        """
        pass


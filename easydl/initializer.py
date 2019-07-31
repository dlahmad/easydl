from typing import Tuple, List
import numpy as np
from .abstract_object import AbstractObject


class Initializer(AbstractObject):

    def init_variable(self, size: List[int]) -> np.ndarray:
        pass


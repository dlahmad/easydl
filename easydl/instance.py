from typing import Union, List
from .abstract_object import AbstractObject
from .abstract_tensor import AbstractTensor
import numpy as np


class Instance(AbstractObject):

    def __init__(self, input_tensors: Union[AbstractTensor, List[AbstractTensor]], output_tensor: AbstractTensor,
                 cache: Union[np.ndarray, List[np.ndarray], None], batch_size: int):
        super().__init__()
        self.input_tensors: Union[AbstractTensor, List[AbstractTensor]] = input_tensors
        self.output_tensor: AbstractTensor = output_tensor
        self.cache: Union[np.ndarray, List[np.ndarray], None] = cache
        self.batch_size = batch_size


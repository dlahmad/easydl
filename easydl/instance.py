from typing import Union, List, Tuple

import numpy as np

from .abstract_object import AbstractObject
from .abstract_tensor import AbstractTensor


class Instance(AbstractObject):
    """
    If there is a forward pass in a node an instance for that particular
    pass is created. It consists of the input tensors, the output tensors
    and the cache of that specific forward pass. If a node is called a second
    time, a new instance is created.
    """
    def __init__(self, input_tensors: Union[AbstractTensor, List[AbstractTensor]], output_tensor: AbstractTensor,
                 cache: Union[np.ndarray, List[np.ndarray], None], batch_size: int):
        """
        When a node is called for one forward pass one instance is created.
        :type batch_size: Batch size of the current instance.
        :type cache: The cache of the forward pass of the current instance.
        :type output_tensor: The output tensors corresponding to the input tensors
        of the current instance.
        :type input_tensors: The input tensors of the current instance.
        """
        super().__init__()
        self.input_tensors: Union[AbstractTensor, List[AbstractTensor]] = input_tensors
        self.output_tensor: AbstractTensor = output_tensor
        self.cache: Union[np.ndarray, List[np.ndarray], None] = cache
        self.batch_size = batch_size


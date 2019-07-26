from typing import Union, List
from .abstract_tensor import AbstractTensor


class Instance:

    def __init__(self, input_tensors: Union[AbstractTensor, List[AbstractTensor]], output_tensor: AbstractTensor):
        self.input_tensors: Union[AbstractTensor, List[AbstractTensor]] = input_tensors
        self.output_tensor: AbstractTensor = output_tensor


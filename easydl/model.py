from typing import Sequence, Union, Tuple
from .abstract_tensor import AbstractTensor
from .abstract_object import AbstractObject


class Model(AbstractObject):
    """
    Base class for building models with the framework. Override the forward
    function to implement the functionality. The backward pass is computed automatically.
    """

    def __call__(self, args: Union[AbstractTensor, Tuple[AbstractTensor]]) -> AbstractTensor:
        """
        Calls the modules forward method.
        :param args: Input tensors for the forward pass.
        :return: Resulting tensor.
        """
        return self.forward(args)

    def forward(self, x: Union[AbstractTensor, Tuple[AbstractTensor]]) -> AbstractTensor:
        """
        Forward pass of the model.
        :param x: Input tensors for the forward pass.
        :return: Resulting tensor.
        """
        pass

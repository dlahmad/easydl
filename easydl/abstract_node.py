from typing import Union, Sequence, Dict, Tuple

import numpy as np

from .abstract_object import AbstractObject
from .util.numerics import get_eps


class AbstractNode(AbstractObject):
    """
    Implements abstract methods and properties needed for
    all node classes creating graphs.
    """

    eps = get_eps()

    def __init__(self):
        super().__init__()

        """Name of the node."""
        self.name: Union[str, None] = self.__class__.__name__

        self.needs_gradient: bool = False
        """Indicates if this node needs gradient optimization."""

        self.propagates_gradient: bool = False
        """Indicates if this node propagates gradients to earlier layers."""

        self.needs_input_check: bool = False
        """If set to true the input_check method is called."""

        self.built: bool = False
        """Indicates if the node has already been built using the
        build method."""

        self.variables: Dict[str, np.ndarray] = {}
        """Contains the variables of the current node."""

        self.gradients: Dict[str, np.ndarray] = {}
        """Contains the gradients for the variables of the current node."""

        self.optimizer_cache: Dict[str, np.ndarray] = {}
        """Contains the cache variables of the current optimizer for the
        variables of the current node."""

    def input_check(self, inputs: Union[np.ndarray, Sequence[np.ndarray]]) -> None:
        """
        This method can be overridden by subclasses and allows for the implementation
        of input checks before the inputs are passed to the forward method. In the case
        of a mal formatted input it should throw an exception. This method is called only
        when the property 'needs_input_check' is set to true.
        :param inputs: Inputs to check.
        :raises Exception: Contains the error in input formatting.
        """
        pass

    def build(self) -> None:
        """
        This method can be overridden by subclasses and allows for the creation of variables
        and other initialization tasks. It is called before the forward method and may be called
        twice if a reinitialization of the node is needed.
        """
        pass

    def forward(self, inputs: Union[np.ndarray, Sequence[np.ndarray]], batch_size: int) -> \
            Tuple[np.ndarray, Union[None, np.ndarray, Sequence[np.ndarray]]]:
        """
        Implements the forward operations of the current node. This method is to be overridden
        by every node.
        :param inputs: Inputs passed to the node.
        :param batch_size: Batch size of the current inputs.
        :return: Operation result and cache which is available to the backward pass.
        """
        pass

    def backward(self, gradients: np.ndarray, cache: Union[None, np.ndarray, Sequence[np.ndarray]], batch_size)\
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Implements the backward operations of the current node. This method is to be overridden
        by every node.
        :param gradients: Gradients for the current node.
        :param cache: Cache of the forward pass corresponding to the current gradients.
        :param batch_size: Batch size of the current pass.
        :return: List of gradients for previous nodes.
        """
        pass

    def raw_backward(self, output_tensor: object, gradients: np.ndarray) -> None:
        """
        This method is executed internally to back-propagate and compute the gradients.
        :param output_tensor: Base tensor to start from.
        :param gradients: Base gradients to start from.
        """
        pass






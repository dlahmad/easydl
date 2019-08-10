from typing import Set, Tuple

from .abstract_node import AbstractNode
from .instance import Instance


class AbstractTape:
    """
    Abstract class for tapes.
    """

    def __init__(self):
        self.operations: Set[Tuple[AbstractNode, Instance]] = set()
        """Set of operations recorded by the tape."""

        self.gradient_operations: Set[Tuple[AbstractNode, Instance]] = set()
        """Set of operations which need gradients recorded by the tape. This is a
        subset of 'operations'."""


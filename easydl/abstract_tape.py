from typing import Set, Tuple
from .abstract_node import AbstractNode
from .instance import Instance


class AbstractTape:

    def __init__(self):
        self.operations: Set[Tuple[AbstractNode, Instance]] = set()
        self.gradient_operations: Set[Tuple[AbstractNode, Instance]] = set()


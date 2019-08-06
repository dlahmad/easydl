from __future__ import annotations
from typing import Set, Tuple
from .abstract_node import AbstractNode
from .instance import Instance


class Tape:

    current_optimizer_set: Set[Tape] = set()

    def __init__(self):
        self.operations: Set[Tuple[AbstractNode, Instance]] = set()
        self.gradient_operations: Set[Tuple[AbstractNode, Instance]] = set()

    def __enter__(self):
        if self is Tape.current_optimizer_set:
            raise Exception("You can only use the same tape once. Please"
                            "leave the 'with' statement of the first usage first.")
        Tape.current_optimizer_set.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tape.current_optimizer_set.remove(self)

    @staticmethod
    def add_node(node: Tuple[AbstractNode, Instance], needs_gradient):
        for tape in Tape.current_optimizer_set:
            tape.operations.add(node)

            if needs_gradient:
                tape.gradient_operations.add(node)

    def clear(self):
        for node, instance in self.operations:
            del node.instances[instance.output_tensor]

        self.operations = set()

    def __del__(self):
        self.clear()

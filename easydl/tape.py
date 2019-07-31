from __future__ import annotations
from typing import Set
from .node import Node


class Tape:

    current_optimizer_set: Set[Tape] = set()

    def __init__(self):
        self.operations: Set[Node] = set()

    def __enter__(self):
        if self is Tape.current_optimizer_set:
            raise Exception("You can only use the same tape once. Please"
                            "leave the 'with' statement of the first usage first.")
        Tape.current_optimizer_set.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tape.current_optimizer_set.remove(self)

    @staticmethod
    def add_node(node: Node):
        for tape in Tape.current_optimizer_set:
            tape.operations.add(node)

    def clear(self):
        self.operations = set()

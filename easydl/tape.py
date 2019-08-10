from typing import Set, Tuple

from .abstract_node import AbstractNode
from .abstract_tape import AbstractTape
from .instance import Instance


class Tape(AbstractTape):

    current_optimizer_set: Set[AbstractTape] = set()

    def __init__(self):
        super().__init__()

    def __enter__(self):
        if self in Tape.current_optimizer_set:
            raise Exception("You can only use the same tape once. Please"
                            "leave the 'with' statement of the first usage first.")
        Tape.current_optimizer_set.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tape.current_optimizer_set.remove(self)

    @staticmethod
    def tape_active() -> bool:
        return len(Tape.current_optimizer_set) > 0

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

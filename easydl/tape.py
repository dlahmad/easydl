from typing import Set, Tuple

from .abstract_node import AbstractNode
from .abstract_tape import AbstractTape
from .instance import Instance


class Tape(AbstractTape):
    """
    The tape allows for the gradient recording while performing operations.
    It can be fed to an optimizer.
    """

    current_optimizer_set: Set[AbstractTape] = set()
    """Contains the tapes which are currently active."""

    def __init__(self):
        """
        Creates a tape for recording node operations.
        """
        super().__init__()

    def __enter__(self):
        if self in Tape.current_optimizer_set:
            raise Exception("You can only use the same tape once. Please"
                            "leave the 'with' statement of the first usage first.")
        Tape.current_optimizer_set.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tape.current_optimizer_set.remove(self)

    def __del__(self):
        self.clear()

    @staticmethod
    def tape_active() -> bool:
        """
        Indicates whether there currently is a tape recording node operations.
        :return: True if is a type recording.
        """
        return len(Tape.current_optimizer_set) > 0

    @staticmethod
    def add_node(node: Tuple[AbstractNode, Instance], needs_gradient) -> None:
        """
        Adds a node on which a operations was performed to all tapes recording.
        :param node: Node the add to all activate tapes.
        :param needs_gradient: Indicates if the node operation performed needs
        an optimization using gradients.
        """
        for tape in Tape.current_optimizer_set:
            tape.operations.add(node)

            if needs_gradient:
                tape.gradient_operations.add(node)

    def clear(self) -> None:
        """
        Deletes gradients and object structure recorded
        while operations were performed.
        """
        for node, instance in self.operations:
            del node.instances[instance.output_tensor]

        self.operations = set()

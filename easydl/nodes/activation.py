from ..node import Node


class Activation(Node):

    def __init__(self):
        super().__init__()
        self.needs_gradient = True


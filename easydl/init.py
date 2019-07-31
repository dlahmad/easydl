from .node import Node
from .tensor import tensor
from .nodes.layer import Layer


def init_easydl():
    Node.tensor_obj = tensor



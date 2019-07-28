from .node import Node
from .tensor import tensor


def init_easydl():
    Node.tensor_obj = tensor


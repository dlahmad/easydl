from .config import Config
from .tensor import tensor


def init_easydl(use_gpu=False) -> None:
    """
    Init needs to be executed BEFORE doing anything
    else withe the framework.
    :param use_gpu: Indicates whether to use cpu or gpu arrays.
    """
    Config.use_gpu = use_gpu
    Config.initialized = True
    Config.tensor_obj = tensor


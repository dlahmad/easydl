from .config import Config
from .tensor import tensor


def init_easydl(use_gpu=False):
    Config.use_gpu = use_gpu
    Config.initialized = True
    Config.tensor_obj = tensor


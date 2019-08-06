from .tensor import tensor
from .config import Config
import numpy as np
import cupy as cp


def init_easydl(use_gpu=False):
    Config.use_gpu = use_gpu
    Config.initialized = True
    Config.tensor_obj = tensor


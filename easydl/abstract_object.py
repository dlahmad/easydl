import numpy as np
import cupy as cp
from .config import Config


class AbstractObject:

    def __init__(self):
        self._use_gpu = Config.use_gpu
        self.np = cp if Config.use_gpu else np

    @property
    def uses_gpu(self):
        return self._use_gpu

    def to_cpu(self):
        if self._use_gpu:
            self.np = np
            for key, val in self.__dict__.items():
                if isinstance(val, AbstractObject):
                    val.to_cpu()
                elif isinstance(val, np.ndarray):
                    self.__dict__[key] = cp.asnumpy(val)
                elif isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, np.ndarray):
                            val[sub_key] = cp.asnumpy(sub_val)
                elif isinstance(val, list):
                    for sub_key, sub_val in enumerate(val):
                        if isinstance(sub_val, np.ndarray):
                            val[sub_key] = cp.asnumpy(sub_val)
            self._use_gpu = False

    def to_gpu(self):
        if not self._use_gpu:
            self.np = cp
            for key, val in self.__dict__.items():
                if isinstance(val, AbstractObject):
                    val.to_gpu()
                elif isinstance(val, np.ndarray):
                    self.__dict__[key] = self.np.asarray(val)
                elif isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, np.ndarray):
                            val[sub_key] = self.np.asarray(sub_val)
                elif isinstance(val, list):
                    for sub_key, sub_val in enumerate(val):
                        if isinstance(sub_val, np.ndarray):
                            val[sub_key] = self.np.asarray(sub_val)

            self._use_gpu = True


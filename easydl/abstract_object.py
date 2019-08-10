import cupy as cp
import numpy as np

from .config import Config


class AbstractObject:
    """
    This is a base class for every other class performing computational operations.
    It defines basic functions for automatic cpu and gpu usage. It also implements functions
    for variable transfer from cpu to gpu and in reverse.
    """

    def __init__(self):
        self._use_gpu: bool = Config.use_gpu
        """Indicates whether the variables used in this object are on cpu or gpu."""

        self.np = cp if Config.use_gpu else np
        """Computational lib used for operations. It can be numpy or cupy depending
        on the device we are working on."""

    @property
    def uses_gpu(self) -> bool:
        """
        Indicates whether the variables used in this object are on cpu or gpu.
        :return: True if variables are on a gpu.
        """
        return self._use_gpu

    def to_cpu(self) -> None:
        """
        Transfers all variables from gpu to cpu variables.
        """
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

    def to_gpu(self) -> None:
        """
        Transfers all variables from cpu to gpu variables.
        """
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


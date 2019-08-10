from typing import Union


class Config:
    """
    The config allows for specific initialization of cpu and gpu models. It
    also contains important framework settings.
    """
    initialized: bool = False
    """Indicates whether the framework was initialized or not."""

    use_gpu: bool = None
    """Indicates whether the framework should use cpu or gpu arrays. (numpy or cupy)"""

    tensor_obj: object = None
    """Hack to prevent circular dependencies of tensor and node class."""

    def __init__(self, use_gpu=False):
        self.temporary_use_gpu: bool = use_gpu
        self.backup_use_gpu: Union[bool, None] = None

    def __enter__(self):
        self.backup_use_gpu = Config.use_gpu
        Config.use_gpu = self.temporary_use_gpu

    def __exit__(self, exc_type, exc_val, exc_tb):
        Config.use_gpu = self.backup_use_gpu


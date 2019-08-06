from typing import Union


class Config:

    initialized: bool = False

    use_gpu: bool = None

    tensor_obj: object = None

    def __init__(self, use_gpu=False):
        self.temporary_use_gpu: bool = use_gpu
        self.backup_use_gpu: Union[bool, None] = None

    def __enter__(self):
        self.backup_use_gpu = Config.use_gpu
        Config.use_gpu = self.temporary_use_gpu

    def __exit__(self, exc_type, exc_val, exc_tb):
        Config.use_gpu = self.backup_use_gpu


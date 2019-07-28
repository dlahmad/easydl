
class Optimizer:

    current_optimizer = None

    def __init__(self):
        self.gradient_dict = {}

    def __enter__(self):
        if Optimizer.current_optimizer is not None:
            raise Exception("You can only use one optimizer. Another one"
                            "is already set")
        Optimizer.current_optimizer = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Optimizer.current_optimizer = None


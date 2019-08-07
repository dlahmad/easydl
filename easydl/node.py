import numpy as np
import gc
from typing import Union, List, Dict, Tuple
from .abstract_node import AbstractNode
from .abstract_tensor import AbstractTensor
from .instance import Instance
from .config import Config
from .util.input_check import check_and_return_batch_size
from .tape import Tape


class Node(AbstractNode):

    def __init__(self):
        super().__init__()
        self.propagates_gradient: bool = True
        self.needs_input_check: bool = True
        self.variables: Dict[str, np.ndarray] = {}
        self.gradients: Dict[str, np.ndarray] = {}
        self.optimizer_cache: Dict[str, np.ndarray] = {}
        self.instances: Dict[AbstractTensor, Instance] = {}
        self.tensor_obj = Config.tensor_obj

    def __call__(self, args: Union[AbstractTensor, List[AbstractTensor]]):
        if self.needs_init and not self.built:
            self.build()
            self.built = True

        inputs = self._to_list(args)
        batch_size = check_and_return_batch_size(inputs)
        numpy_inputs = list(map(self._tensor2numpy, inputs))

        if self.needs_input_check:
            self.input_check(numpy_inputs)

        numpy_output, cache = self.forward(numpy_inputs, batch_size)
        output = self._numpy2tensor(numpy_output)

        if Tape.tape_active():
            instance = Instance(inputs, output, cache, batch_size)
            self.instances[output] = instance
            Tape.add_node((self, instance), self.needs_gradient)

        return output

    def raw_backward(self, output_tensor: AbstractTensor, gradients: np.ndarray) -> None:
        if output_tensor not in self.instances:
            raise Exception('Could not find an instance which created this tensor!'
                            ' Did you use a tape to record the operations before calling "backwards"?')

        base_instance = self.instances[output_tensor]
        graph = self._build_dynamic_graph(base_instance)

        level_grads = {output_tensor: gradients}

        instance: Instance
        for level in graph.values():
            new_level_grads = {}

            for instance in level:
                current_node = instance.output_tensor.origin
                input_tensors = instance.input_tensors
                input_grads = current_node.backward(level_grads[instance.output_tensor]
                                                    , instance.cache, instance.batch_size)
                if not isinstance(input_grads, (list, tuple)):
                    input_grads = list([input_grads])

                for inp, grad in zip(input_tensors, input_grads):
                    new_level_grads[inp] = grad

            level_grads = new_level_grads

    @staticmethod
    def _build_dynamic_graph(base_instance: Instance):
        hierarchy = {0: [base_instance]}

        level = 0
        while level < 1000 and hierarchy[level]:
            current_level = []

            for instance in hierarchy[level]:
                input_tensors = instance.input_tensors
                if not isinstance(input_tensors, list):
                    input_tensors = list([input_tensors])
                for input_tensor in input_tensors:
                    if input_tensor.origin:
                        origin_instance = input_tensor.origin.instances[input_tensor]
                        current_level.append(origin_instance)
            level += 1
            hierarchy[level] = current_level

        if level >= 1000:
            raise Exception('Graph to deep! Is has thousand or more levels!')

        del hierarchy[level]

        return hierarchy

    @staticmethod
    def _tensor2numpy(inp: AbstractTensor) -> np.ndarray:
        return inp.numpy

    def _numpy2tensor(self, inp: np.ndarray, set_origin=True) -> AbstractTensor:
        tns = self.tensor_obj(inp)

        if set_origin:
            tns.register_origin(self)

        return tns

    @staticmethod
    def _to_list(var):
        if not isinstance(var, list):
            return [var]
        return var






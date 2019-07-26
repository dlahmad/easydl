import numpy as np
from typing import Union, List, Dict, Tuple
from .abstract_node import AbstractNode
from .abstract_tensor import AbstractTensor
from .instance import Instance
#from .nodes.constant import Constant
#from .nodes.add import Add
import easydl.nodes as nodes
import sys, inspect


class Node(AbstractNode):

    tensor_obj = None

    def __init__(self):
        super().__init__()
        self.variables: Dict[str, np.ndarray] = {}
        self.instances: Dict[AbstractTensor, Instance] = {}
        self.node_type_dict = {}

    def __call__(self, args: Union[AbstractTensor, List[AbstractTensor]]):
        if not self.built:
            self.build()

        inputs = args
        numpy_inputs = list(map(self._tensor2numpy, inputs))

        self.input_check(numpy_inputs)

        numpy_output = self.forward(numpy_inputs)
        output = self._numpy2tensor(numpy_output)

        instance = Instance(inputs, output)
        self.instances[output] = instance

        return output

    def _internal_backward(self, output_tensor: AbstractTensor, gradients: np.ndarray) -> None:
        base_instance = self.instances[output_tensor]
        graph = self._build_dynamic_graph(base_instance)

        level_grads = {output_tensor: gradients}

        instance: Instance
        for level in graph.values():
            new_level_grads = {}

            for instance in level:
                numpy_inputs = list(map(self._tensor2numpy, instance.input_tensors))
                input_grads = instance.output_tensor.origin.backward(level_grads[instance.output_tensor], numpy_inputs)

                new_level_grads.update(dict(zip(instance.input_tensors, input_grads)))

                for inp, grad in zip(instance.input_tensors, input_grads):
                    inp.grad += grad

            level_grads = new_level_grads

    @staticmethod
    def _build_dynamic_graph(base_instance: Instance):
        hierarchy = {0: [base_instance]}

        level = 0
        while level < 1000 and hierarchy[level]:
            current_level = []

            for instance in hierarchy[level]:
                for input_tensor in instance.input_tensors:
                    if input_tensor.origin:
                        current_level.append(input_tensor.origin.instances[input_tensor])
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

from typing import Union, List, Dict

import numpy as np

from .abstract_node import AbstractNode
from .abstract_tensor import AbstractTensor
from .config import Config
from .instance import Instance
from .tape import Tape
from .util.input_check import check_and_return_batch_size


class Node(AbstractNode):
    """
    Base object for all nodes.
    """

    def __init__(self):
        """
        Creates a node object.
        """

        super().__init__()
        self.propagates_gradient: bool = True
        """Indicates if this node propagates gradients to earlier layers."""

        self.needs_input_check: bool = True
        """If set to true the input_check method is called."""

        self.instances: Dict[AbstractTensor, Instance] = {}
        """Contains the instances which 'passed' through this node."""

        self.tensor_obj = Config.tensor_obj
        """Contains the class which can create tensor objects. This is a hack
        to prevent circle dependencies."""

    def __call__(self, args: Union[AbstractTensor, List[AbstractTensor]]):
        """
        Overwrites the call operator and allows for the execution of node operations.
        This method calls the forward method and creates the resulting tensor. It also
        saves the particular forward pass to one particular instance.
        :param args: Args provided to the node.
        :return: Result tensor.
        """
        if not self.built:
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
        """
        This method is executed internally to back-propagate and compute the gradients.
        :param output_tensor: Base tensor to start from.
        :param gradients: Base gradients to start from.
        """
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
    def _build_dynamic_graph(base_instance: Instance) -> Dict[int, List[Instance]]:
        """
        This method creates the dynamic backward graph based
        on one starting instance. This hierarchy is used to compute
        the backward pass and guarantees that all grads required from
        later layers are always available to backward operation they
        are needed in.
        :param base_instance: Instance to start from.
        :return: Hierarchical level graph for backwards iteration.
        """
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
        """
        Converts a tensor to an numpy or cupy array.
        :param inp: Tensor to convert.
        :return: Numpy or cupy array.
        """
        return inp.numpy

    def _numpy2tensor(self, inp: np.ndarray, set_origin=True) -> AbstractTensor:
        """
        Converts a numpy or cupy array to a tensor. This uses a slight hack to prevent
        circle dependencies.
        :param inp: Numpy or cupy array.
        :param set_origin: Indicates whether to set this note as the tensor origin or not.
        :return: Created tensor.
        """
        tns = self.tensor_obj(inp)

        if set_origin:
            tns.register_origin(self)

        return tns

    @staticmethod
    def _to_list(var):
        """
        Checks whether a variable is a list or not. If not is creates
        a single element list containing the variable.
        :param var: Variable to check.
        :return: List which was passed or created.
        """
        if not isinstance(var, list):
            return [var]
        return var






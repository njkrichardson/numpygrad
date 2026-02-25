from numpygrad.core.array import Array
from numpygrad.nn.module import Module
from numpygrad.nn.neuron import Neuron


class Linear(Module):
    def __init__(self, num_inputs: int, num_outputs: int, **kwargs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

    def __call__(self, x: list[Array] | Array) -> list[Array] | Array:
        out = [neuron(x) for neuron in self.neurons]
        if len(out) == 1:
            return out[0]
        return out

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_inputs={self.num_inputs}, num_outputs={self.num_outputs})"

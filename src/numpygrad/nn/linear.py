import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.array_creation import randn, zeros
from numpygrad.nn.module import Module


class Linear(Module):
    def __init__(self, num_inputs: int, num_outputs: int, **kwargs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight = Array(
            np.random.randn(num_outputs, num_inputs) * np.sqrt(2 / num_inputs),
            requires_grad=True,
        )
        self.bias = zeros(num_outputs, requires_grad=True)

    def __call__(self, x: Array) -> Array:
        return (x @ self.weight.T) + self.bias

    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_inputs={self.num_inputs}, num_outputs={self.num_outputs})"

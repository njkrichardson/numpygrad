import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.array_creation import zeros
from numpygrad.nn.init import xavier_normal_
from numpygrad.nn.module import Module, Parameter


class Linear(Module):
    def __init__(self, num_inputs: int, num_outputs: int, **kwargs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight = Parameter(
            xavier_normal_(Array(np.empty((num_inputs, num_outputs)), requires_grad=True))
        )
        self.bias = Parameter(zeros(num_outputs, requires_grad=True))

    def forward(self, x: Array) -> Array:
        return (x @ self.weight) + self.bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_inputs={self.num_inputs}, \
        num_outputs={self.num_outputs}, bias={self.bias is not None})"

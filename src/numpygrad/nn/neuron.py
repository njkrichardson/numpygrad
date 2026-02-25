from numpygrad.nn.module import Module
from numpygrad.core.array import Array
from numpygrad.core.array_creation import randn, zeros


class Neuron(Module):
    def __init__(self, num_inputs: int, activate: bool = True):
        self.weight = [randn() for _ in range(num_inputs)]
        self.bias = zeros(1)
        self.activate = activate

    def __call__(self, x: list[Array] | Array) -> Array:
        if isinstance(x, Array):
            x = [x]
        activation = sum((x_i * w_i for x_i, w_i in zip(x, self.weight)), self.bias)
        if self.activate:
            return activation.relu()
        return activation

    def parameters(self):
        return self.weight + [self.bias]

    def __repr__(self) -> str:
        return f"Neuron(weight={self.weight}, bias={self.bias})"

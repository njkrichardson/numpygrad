from numpygrad.nn.module import Module
from numpygrad.nn.linear import Linear
from numpygrad.core.array import Array


class MLP(Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int):
        dims = [input_dim] + hidden_sizes + [output_dim]
        self.layers = [Linear(nin, nout) for nin, nout in zip(dims[:-1], dims[1:])]

    def parameters(self) -> list[Array]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([repr(layer) for layer in self.layers])})"

    def __call__(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)

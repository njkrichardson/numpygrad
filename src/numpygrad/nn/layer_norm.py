import numpy as np

from numpygrad.core.array import Array
from numpygrad.nn.module import Module, Parameter


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(Array(np.ones(normalized_shape)))
            self.bias = Parameter(Array(np.zeros(normalized_shape)))

    def forward(self, x: Array) -> Array:
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(axis=axis, keepdims=True)
        var = x.var(axis=axis, keepdims=True)
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm

    def __repr__(self) -> str:
        return (
            f"LayerNorm({self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine})"
        )

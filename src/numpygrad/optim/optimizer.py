import numpy as np

from numpygrad.core.array import Array


class Optimizer:
    def __init__(self, params: list[Array]):
        self.params = params

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = np.zeros_like(param.data)

    def step(self) -> None:
        raise NotImplementedError

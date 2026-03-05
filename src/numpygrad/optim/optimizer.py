import numpy as np

from numpygrad.nn.module import Parameter


class Optimizer:
    def __init__(self, params: list[Parameter]):
        self.params = params

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = np.zeros_like(param.data)

    def state_dict(self) -> list[dict]:
        return []

    def load_state_dict(self, state: list[dict]) -> None:
        pass

    def step(self) -> None:
        raise NotImplementedError

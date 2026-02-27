from numpygrad.optim.optimizer import Optimizer
from numpygrad.core.array import Array


class SGD(Optimizer):
    def __init__(self, params: list[Array], step_size: float = 1e-3):
        super().__init__(params)
        self.step_size = step_size

    def step(self) -> None:
        for param in self.params:
            assert param.grad is not None
            param.data -= self.step_size * param.grad

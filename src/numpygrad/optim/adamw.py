import numpy as np

from numpygrad.core.array import Array
from numpygrad.optim.optimizer import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: list[Array],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._state: dict[int, dict] = {}  # per-param: {t, m, v}

    def step(self) -> None:
        b1, b2 = self.betas
        for param in self.params:
            assert param.grad is not None
            pid = id(param)
            if pid not in self._state:
                self._state[pid] = {
                    "t": 0,
                    "m": np.zeros_like(param.data),
                    "v": np.zeros_like(param.data),
                }
            state = self._state[pid]
            state["t"] += 1
            t = state["t"]
            g = param.grad
            state["m"] = b1 * state["m"] + (1 - b1) * g
            state["v"] = b2 * state["v"] + (1 - b2) * g**2
            m_hat = state["m"] / (1 - b1**t)
            v_hat = state["v"] / (1 - b2**t)
            # Decoupled weight decay
            param.data *= 1 - self.lr * self.weight_decay
            # Gradient step
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

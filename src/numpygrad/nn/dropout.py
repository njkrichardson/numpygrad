import numpy as np

import numpygrad as npg
from numpygrad.core.array import Array
from numpygrad.nn.module import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        object.__setattr__(self, "p", p)

    def forward(self, x: Array) -> Array:
        if not self.training or self.p == 0.0:
            return x
        keep = np.random.rand(*x.shape) >= self.p
        mask = keep.astype(x.data.dtype) / (1.0 - self.p)
        return x * npg.array(mask)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"

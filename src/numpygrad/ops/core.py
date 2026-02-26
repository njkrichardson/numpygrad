import numpy as np

from numpygrad.core.array import Array, ArrayCoercible


def ensure_array(x: ArrayCoercible) -> Array:
    if isinstance(x, Array):
        return x
    elif isinstance(x, np.ndarray):
        return Array(x)
    else:
        return Array(np.array(x))

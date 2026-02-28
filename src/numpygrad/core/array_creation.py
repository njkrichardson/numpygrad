import numpy as np
import numpy.random as npr

import numpygrad as npg
from numpygrad.core.array import Array


def ones(shape: tuple[int, ...] | int, **kwargs) -> Array:
    return npg.array(np.ones(shape), **kwargs)


def zeros(shape: tuple[int, ...] | int, **kwargs) -> npg.ndarray:
    return npg.array(np.zeros(shape), **kwargs)


def zeros_like(x: npg.ndarray | np.ndarray, **kwargs) -> npg.ndarray:
    return npg.array(np.zeros_like(x.data if isinstance(x, npg.array) else x), **kwargs)


def arange(start: float, stop: float | None = None, step: int = 1, **kwargs) -> npg.ndarray:
    if stop is None:
        stop = start
        start = 0
    return npg.array(np.arange(start, stop, step), **kwargs)


def linspace(start: float, stop: float, num: int) -> npg.ndarray:
    return npg.array(np.linspace(start, stop, num))


def eye(n: int) -> npg.ndarray:
    return npg.array(np.eye(n))


def randn(shape: tuple[int, ...] | int = (1,), **kwargs) -> npg.ndarray:
    if isinstance(shape, int):
        shape = (shape,)
    return npg.array(npr.randn(*shape), **kwargs)


def randint(
    low: int, high: int | None = None, size: tuple[int, ...] | int = (1,), **kwargs
) -> npg.ndarray:
    if high is None:
        high = low
        low = 0
    if isinstance(size, int):
        size = (size,)
    return npg.array(npr.randint(low, high, size), **kwargs)

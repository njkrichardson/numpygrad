import numpy as np

from numpygrad.core.array import Array


def ones(shape: tuple[int, ...] | int, **kwargs) -> Array:
    return Array(np.ones(shape), **kwargs)


def zeros(shape: tuple[int, ...] | int, **kwargs) -> Array:
    return Array(np.zeros(shape), **kwargs)


def empty(shape: tuple[int, ...] | int, **kwargs) -> Array:
    return Array(np.empty(shape), **kwargs)


def zeros_like(x: Array | np.ndarray, **kwargs) -> Array:
    return Array(np.zeros_like(x.data if isinstance(x, Array) else x), **kwargs)


def arange(start: float, stop: float | None = None, step: int = 1, **kwargs) -> Array:
    if stop is None:
        stop = start
        start = 0
    return Array(np.arange(start, stop, step), **kwargs)


def linspace(start: float, stop: float, num: int) -> Array:
    return Array(np.linspace(start, stop, num))


def eye(n: int) -> Array:
    return Array(np.eye(n))

import numpy.random as npr

from numpygrad.core.array import Array
from numpygrad.core.random import manual_seed as manual_seed  # re-export


def rand(shape: tuple[int, ...] | int = (1,), **kwargs) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return Array(npr.rand(*shape), **kwargs)


def randn(shape: tuple[int, ...] | int = (1,), **kwargs) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return Array(npr.randn(*shape), **kwargs)


def randint(
    low: int, high: int | None = None, size: tuple[int, ...] | int = (1,), **kwargs
) -> Array:
    if high is None:
        high = low
        low = 0
    if isinstance(size, int):
        size = (size,)
    return Array(npr.randint(low, high, size), **kwargs)


def uniform(
    low: float = 0.0, high: float = 1.0, size: tuple[int, ...] | int = (1,), **kwargs
) -> Array:
    if isinstance(size, int):
        size = (size,)
    return Array(npr.uniform(low, high, size), **kwargs)


def normal(
    mean: float = 0.0, std: float = 1.0, size: tuple[int, ...] | int = (1,), **kwargs
) -> Array:
    if isinstance(size, int):
        size = (size,)
    return Array(npr.normal(mean, std, size), **kwargs)


def randperm(n: int, **kwargs) -> Array:
    return Array(npr.permutation(n), **kwargs)

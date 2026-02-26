from numpygrad.core.array import Array
import numpygrad as npg
import numpy as np
import numpy.random as npr


def ones(shape: tuple[int, ...] | int, **kwargs) -> Array:
    return npg.array(np.ones(shape), **kwargs)


def zeros(shape: tuple[int, ...] | int, **kwargs) -> npg.ndarray:
    return npg.array(np.zeros(shape), **kwargs)


def zeros_like(x: npg.ndarray | np.ndarray, **kwargs) -> npg.ndarray:
    return npg.array(
        np.zeros_like(x.data if isinstance(x, npg.ndarray) else x), **kwargs
    )


def arange(start: float, stop: float, step: float) -> npg.ndarray:
    return npg.array(np.arange(start, stop, step))


def linspace(start: float, stop: float, num: int) -> npg.ndarray:
    return npg.array(np.linspace(start, stop, num))


def eye(n: int) -> npg.ndarray:
    return npg.array(np.eye(n))


def randn(shape: tuple[int, ...] = (1,), **kwargs) -> npg.ndarray:
    return npg.array(npr.randn(*shape), **kwargs)

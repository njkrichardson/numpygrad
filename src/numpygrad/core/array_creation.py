import numpygrad as npg
import numpy as np
import numpy.random as npr


def ones(shape: tuple[int, ...] | int) -> npg.ndarray:
    return npg.array(np.ones(shape))


def zeros(shape: tuple[int, ...] | int) -> npg.ndarray:
    return npg.array(np.zeros(shape))


def zeros_like(x: npg.ndarray | np.ndarray) -> npg.ndarray:
    return npg.array(np.zeros_like(x.data if isinstance(x, npg.ndarray) else x))


def arange(start: float, stop: float, step: float) -> npg.ndarray:
    return npg.array(np.arange(start, stop, step))


def linspace(start: float, stop: float, num: int) -> npg.ndarray:
    return npg.array(np.linspace(start, stop, num))


def eye(n: int) -> npg.ndarray:
    return npg.array(np.eye(n))


def randn(shape: tuple[int, ...] = (1,)) -> npg.ndarray:
    return npg.array(npr.randn(*shape))

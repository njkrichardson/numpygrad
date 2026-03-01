import numpy as np


class VersionCounter:
    __slots__ = ("_v",)

    def __init__(self) -> None:
        self._v = 0

    def increment(self) -> None:
        self._v += 1

    @property
    def version(self) -> int:
        return self._v


def _root(data: np.ndarray) -> np.ndarray:
    while data.base is not None:
        data = data.base
    return data


def _shares_buffer(a: np.ndarray, b: np.ndarray) -> bool:
    return _root(a) is _root(b)

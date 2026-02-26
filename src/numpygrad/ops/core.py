from typing import TypeAlias

import numpy as np

from numpygrad.core.array import Array

ArrayCoercible: TypeAlias = (
    np.ndarray | int | float | list[int | float] | tuple[int | float] | Array
)


def ensure_array(x: ArrayCoercible) -> Array:
    if isinstance(x, Array):
        return x
    elif isinstance(x, np.ndarray):
        return Array(x)
    else:
        return Array(np.array(x))


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    Reduce a gradient array `grad` to match `target_shape` by summing over
    broadcasted dimensions.
    """
    grad_shape = grad.shape
    ndim_diff = len(grad_shape) - len(target_shape)

    # prepend ones to target_shape to align dimensions
    shape_aligned = (1,) * ndim_diff + target_shape

    # sum over axes where target_shape has 1
    axes = tuple(
        i for i, (_, t_dim) in enumerate(zip(grad_shape, shape_aligned)) if t_dim == 1
    )
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    # squeeze any extra leading dimensions
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0, keepdims=False)

    # final sanity check
    assert grad.shape == target_shape, (
        f"unbroadcast failed: got {grad.shape}, expected {target_shape}"
    )

    return grad

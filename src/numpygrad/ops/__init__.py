from numpygrad.core.opid import OperatorId
from numpygrad.core.dispatch import dispatch
from numpygrad.core.array import Array
from numpygrad.ops.core import ArrayCoercible

import numpygrad.ops.comparators as comparators
import numpygrad.ops.elementwise as elementwise
import numpygrad.ops.reductions as reductions
import numpygrad.ops.linalg as linalg
import numpygrad.ops.transforms as transforms


def add(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.ADD, a, b)


def mul(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MUL, a, b)


def sum(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.SUM, a, axis=axis, keepdims=keepdims)


def mm(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MATMUL, a, b)


def relu(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.RELU, a)


def transpose(a: ArrayCoercible, axes: tuple[int, ...]) -> Array:
    return dispatch(OperatorId.TRANSPOSE, a, axes=axes)


def reshape(a: ArrayCoercible, new_shape: tuple[int, ...] | int) -> Array:
    return dispatch(OperatorId.RESHAPE, a, new_shape=new_shape)


def mean(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.MEAN, a, axis=axis, keepdims=keepdims)


matmul = mm


__all__ = [
    "add",
    "mul",
    "elementwise",
    "reductions",
    "sum",
    "linalg",
    "mm",
    "matmul",
    "relu",
    "transpose",
    "transforms",
    "mean",
    "reshape",
]

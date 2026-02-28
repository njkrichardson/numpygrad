import numpygrad.ops.elementwise as elementwise
import numpygrad.ops.linalg as linalg
import numpygrad.ops.reductions as reductions
import numpygrad.ops.special as special
import numpygrad.ops.transforms as transforms
from numpygrad.core.array import Array
from numpygrad.core.dispatch import dispatch
from numpygrad.core.opid import OperatorId
from numpygrad.ops.core import ArrayCoercible

# elementwise


def add(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.ADD, a, b)


def mul(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MUL, a, b)


def exp(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.EXP, a)


def pow(a: ArrayCoercible, power: ArrayCoercible) -> Array:
    return dispatch(OperatorId.POW, a, power)


def log(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.LOG, a)


def abs(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.ABS, a)


def clip(a: ArrayCoercible, min: ArrayCoercible, max: ArrayCoercible) -> Array:
    return dispatch(OperatorId.CLIP, a, min, max)


def maximum(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MAXIMUM, a, b)


def minimum(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MINIMUM, a, b)


def relu(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.RELU, a)


# reductions


def mean(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.MEAN, a, axis=axis, keepdims=keepdims)


def sum(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.SUM, a, axis=axis, keepdims=keepdims)


def prod(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.PRODUCT, a, axis=axis, keepdims=keepdims)


def min(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.MIN, a, axis=axis, keepdims=keepdims)


def max(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.MAX, a, axis=axis, keepdims=keepdims)


def argmax(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.ARGMAX, a, axis=axis, keepdims=keepdims)


# transforms


def transpose(a: ArrayCoercible, axes: tuple[int, ...]) -> Array:
    return dispatch(OperatorId.TRANSPOSE, a, axes=axes)


def unsqueeze(a: ArrayCoercible, axis: int) -> Array:
    return dispatch(OperatorId.UNSQUEEZE, a, axis=axis)


def reshape(a: ArrayCoercible, new_shape: tuple[int, ...] | int) -> Array:
    return dispatch(OperatorId.RESHAPE, a, new_shape=new_shape)


def flatten(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.FLATTEN, a)


def stack(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    return dispatch(OperatorId.STACK, arrays=arrays, axis=axis)


def cat(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    return dispatch(OperatorId.CAT, arrays=arrays, axis=axis)


# linear algebra


def mm(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MATMUL, a, b)


def dot(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.DOT, a, b)


def norm(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    return dispatch(OperatorId.NORM, a, axis=axis, keepdims=keepdims)


matmul = mm

# special methods


def setitem(a: ArrayCoercible, key: tuple[int, ...], value: ArrayCoercible) -> Array:
    return dispatch(OperatorId.SETITEM, a, key, value)


__all__ = [
    # special methods
    "special",
    "setitem",
    # elementwise
    "elementwise",
    "add",
    "mul",
    "sum",
    "relu",
    "exp",
    "pow",
    "log",
    "abs",
    "clip",
    "maximum",
    "minimum",
    # transforms
    "transpose",
    "unsqueeze",
    "reshape",
    "flatten",
    "stack",
    "cat",
    "transforms",
    # reductions
    "reductions",
    "mean",
    "prod",
    # linear algebra
    "linalg",
    "mm",
    "matmul",
    "dot",
    "norm",
]

import numpygrad.ops.activations as activations
import numpygrad.ops.conv as conv
import numpygrad.ops.elementwise as elementwise
import numpygrad.ops.linalg as linalg
import numpygrad.ops.reductions as reductions
import numpygrad.ops.special as special
import numpygrad.ops.transforms as transforms
from numpygrad.core.array import Array
from numpygrad.core.dispatch import dispatch
from numpygrad.core.opid import OperatorId
from numpygrad.ops.core import ArrayCoercible

# activations


def softmax(a: ArrayCoercible, axis: int = -1) -> Array:
    return dispatch(OperatorId.SOFTMAX, a, axis=axis)


def log_softmax(a: ArrayCoercible, axis: int = -1) -> Array:
    return dispatch(OperatorId.LOG_SOFTMAX, a, axis=axis)


def sigmoid(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.SIGMOID, a)


def tanh(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.TANH, a)


def softplus(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.SOFTPLUS, a)


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


def var(a: ArrayCoercible, axis=None, ddof: int = 0, keepdims: bool = False) -> Array:
    return dispatch(OperatorId.VAR, a, axis=axis, ddof=ddof, keepdims=keepdims)


def std(a: ArrayCoercible, axis=None, ddof: int = 0, keepdims: bool = False) -> Array:
    return var(a, axis=axis, ddof=ddof, keepdims=keepdims) ** 0.5


def sqrt(a: ArrayCoercible) -> Array:
    from numpygrad.core.array import Array as _Array

    arr = a if isinstance(a, _Array) else _Array(a)
    return arr**0.5


def cumsum(a: ArrayCoercible, axis=None) -> Array:
    return dispatch(OperatorId.CUMSUM, a, axis=axis)


def cumprod(a: ArrayCoercible, axis=None) -> Array:
    return dispatch(OperatorId.CUMPROD, a, axis=axis)


# transforms


def transpose(a: ArrayCoercible, axes: tuple[int, ...]) -> Array:
    return dispatch(OperatorId.TRANSPOSE, a, axes=axes)


def unsqueeze(a: ArrayCoercible, axis: int) -> Array:
    return dispatch(OperatorId.UNSQUEEZE, a, axis=axis)


def reshape(a: ArrayCoercible, new_shape: tuple[int, ...] | int) -> Array:
    return dispatch(OperatorId.RESHAPE, a, new_shape=new_shape)


def flatten(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.FLATTEN, a)


def squeeze(a: ArrayCoercible, axis=None) -> Array:
    return dispatch(OperatorId.SQUEEZE, a, axis=axis)


def repeat(a: ArrayCoercible, repeats: int, axis=None) -> Array:
    return dispatch(OperatorId.REPEAT, a, repeats=repeats, axis=axis)


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


def diagonal(a: ArrayCoercible, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Array:
    return dispatch(OperatorId.DIAGONAL, a, offset=offset, axis1=axis1, axis2=axis2)


def trace(a: ArrayCoercible, offset: int = 0) -> Array:
    from numpygrad.core.array import Array as _Array

    arr = a if isinstance(a, _Array) else _Array(a)
    return arr.diagonal(offset=offset).sum()


matmul = mm


# convolution


def conv2d(
    input: ArrayCoercible,
    weight: ArrayCoercible,
    bias: ArrayCoercible | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Array:
    _s: tuple[int, int] = (stride, stride) if isinstance(stride, int) else stride
    _p: tuple[int, int] = (padding, padding) if isinstance(padding, int) else padding
    return dispatch(OperatorId.CONV2D, input, weight, bias, _s, _p)


# special methods


def setitem(a: ArrayCoercible, key: tuple[int, ...], value: ArrayCoercible) -> Array:
    return dispatch(OperatorId.SETITEM, a, key, value)


__all__ = [
    # activations
    "activations",
    "softmax",
    "log_softmax",
    "sigmoid",
    "tanh",
    "softplus",
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
    "squeeze",
    "repeat",
    "stack",
    "cat",
    "transforms",
    # reductions
    "reductions",
    "mean",
    "prod",
    "var",
    "std",
    "sqrt",
    "cumsum",
    "cumprod",
    # linear algebra
    "linalg",
    "mm",
    "matmul",
    "dot",
    "norm",
    "diagonal",
    "trace",
    # convolution
    "conv",
    "conv2d",
]

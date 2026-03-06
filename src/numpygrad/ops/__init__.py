import numpy as np

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


def gelu(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.GELU, a)


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


def sin(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.SIN, a)


def cos(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.COS, a)


def tan(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.TAN, a)


def floor(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.FLOOR, a)


def ceil(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.CEIL, a)


def sign(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.SIGN, a)


def copy(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.COPY, a)


def where(cond: ArrayCoercible, x: ArrayCoercible, y: ArrayCoercible) -> Array:
    return dispatch(OperatorId.WHERE, cond, x, y)


def isnan(a: ArrayCoercible) -> Array:
    from numpygrad.ops.core import ensure_array

    return Array(np.isnan(ensure_array(a).data))


def isinf(a: ArrayCoercible) -> Array:
    from numpygrad.ops.core import ensure_array

    return Array(np.isinf(ensure_array(a).data))


def isfinite(a: ArrayCoercible) -> Array:
    from numpygrad.ops.core import ensure_array

    return Array(np.isfinite(ensure_array(a).data))


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


def argmin(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.ARGMIN, a, axis=axis, keepdims=keepdims)


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


def transpose(a: ArrayCoercible, *axes: int | tuple[int, ...]) -> Array:
    axes_ = tuple(axes[0]) if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
    return dispatch(OperatorId.TRANSPOSE, a, axes=axes_)


def permute(a: ArrayCoercible, *dims: int | tuple[int, ...]) -> Array:
    return transpose(a, *dims)


def unsqueeze(a: ArrayCoercible, axis: int) -> Array:
    return dispatch(OperatorId.UNSQUEEZE, a, axis=axis)


def reshape(a: ArrayCoercible, *new_shape: int | tuple[int, ...]) -> Array:
    shape = (
        new_shape[0]
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list, int))
        else new_shape
    )
    return dispatch(OperatorId.RESHAPE, a, new_shape=shape)


def flatten(a: ArrayCoercible) -> Array:
    return dispatch(OperatorId.FLATTEN, a)


def squeeze(a: ArrayCoercible, axis=None) -> Array:
    return dispatch(OperatorId.SQUEEZE, a, axis=axis)


def repeat(a: ArrayCoercible, repeats: int, axis=None) -> Array:
    return dispatch(OperatorId.REPEAT, a, repeats=repeats, axis=axis)


def embedding(weight: ArrayCoercible, indices: ArrayCoercible) -> Array:
    return dispatch(OperatorId.EMBEDDING, weight, indices)


def stack(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    return dispatch(OperatorId.STACK, arrays=arrays, axis=axis)


def cat(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    return dispatch(OperatorId.CAT, arrays=arrays, axis=axis)


def triu(a: ArrayCoercible, k: int = 0) -> Array:
    return dispatch(OperatorId.TRIU, a, k=k)


def split(
    a: ArrayCoercible, split_size_or_sections: int | list[int], dim: int = 0
) -> tuple[Array, ...]:
    return transforms.split(a, split_size_or_sections, dim=dim)


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
concatenate = cat
expand_dims = unsqueeze


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


def masked_fill(a: ArrayCoercible, mask: ArrayCoercible, value: float | int) -> Array:
    return dispatch(OperatorId.MASKED_FILL, a, mask, value)


__all__ = [
    # activations
    "activations",
    "softmax",
    "log_softmax",
    "sigmoid",
    "tanh",
    "softplus",
    "gelu",
    # special methods
    "special",
    "setitem",
    "masked_fill",
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
    "sin",
    "cos",
    "tan",
    "floor",
    "ceil",
    "sign",
    "copy",
    "where",
    "isnan",
    "isinf",
    "isfinite",
    # transforms
    "transpose",
    "permute",
    "unsqueeze",
    "reshape",
    "flatten",
    "squeeze",
    "repeat",
    "embedding",
    "stack",
    "cat",
    "triu",
    "split",
    "transforms",
    # aliases
    "concatenate",
    "expand_dims",
    # reductions
    "reductions",
    "mean",
    "prod",
    "var",
    "std",
    "sqrt",
    "cumsum",
    "cumprod",
    "argmin",
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

from numpygrad.core.opid import OperatorId
from numpygrad.core.dispatch import dispatch
from numpygrad.core.array import Array
from numpygrad.ops.core import ArrayCoercible

import numpygrad.ops.elementwise as elementwise
import numpygrad.ops.reductions as reductions
import numpygrad.ops.linalg as linalg


def add(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.ADD, a, b)


def mul(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MUL, a, b)


def sum(a: ArrayCoercible, axis=None, keepdims=False) -> Array:
    return dispatch(OperatorId.SUM, a, axis=axis, keepdims=keepdims)


def mm(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return dispatch(OperatorId.MATMUL, a, b)


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
]

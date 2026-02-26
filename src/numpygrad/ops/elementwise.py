import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.registry import register, OperatorRequirements
from numpygrad.core.opid import OperatorId
from numpygrad.core.function import Function, Context
from numpygrad.ops.core import ArrayCoercible, ensure_array, unbroadcast


@register(OperatorId.MUL)
def mul_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    a, b = ensure_array(a), ensure_array(b)
    return Array(
        a.data * b.data,
        device="cpu_np",
        requires_grad=False,
    )


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, b: ArrayCoercible) -> Array:
        a, b = ensure_array(a), ensure_array(b)
        ctx.store(a, b)
        return Array(
            a.data * b.data,
            device=a.device,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a, b = ctx.saved_arrays
        return b.data * grad, a.data * grad


@register(OperatorId.MUL, op_requirements=OperatorRequirements.Autograd)
def mul_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Mul.apply(a, b)


@register(OperatorId.ADD)
def add_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    a, b = ensure_array(a), ensure_array(b)
    return Array(
        a.data + b.data,
        device="cpu_np",
        requires_grad=False,
    )


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, b: ArrayCoercible) -> Array:
        a, b = ensure_array(a), ensure_array(b)
        ctx.store(a, b)
        return Array(
            a.data + b.data,
            device=a.device,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a, b = ctx.saved_arrays
        agrad = unbroadcast(grad, a.shape)
        bgrad = unbroadcast(grad, b.shape)
        return agrad, bgrad


@register(OperatorId.ADD, op_requirements=OperatorRequirements.Autograd)
def add_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Add.apply(a, b)

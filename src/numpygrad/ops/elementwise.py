import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.registry import register, OperatorRequirements
from numpygrad.core.opid import OperatorId
from numpygrad.core.function import Function, Context
from numpygrad.ops.core import ensure_array
from numpygrad.core.broadcasting import unbroadcast


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
        agrad = unbroadcast(grad * b.data, a.shape)
        bgrad = unbroadcast(grad * a.data, b.shape)
        return agrad, bgrad


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
        return unbroadcast(grad, a.shape), unbroadcast(grad, b.shape)


@register(OperatorId.ADD, op_requirements=OperatorRequirements.Autograd)
def add_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Add.apply(a, b)


@register(OperatorId.POW)
def pow_cpu(a: ArrayCoercible, power: ArrayCoercible) -> Array:
    a, power = ensure_array(a), ensure_array(power)
    return Array(
        a.data**power.data,
        device="cpu_np",
        requires_grad=False,
    )


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, power: ArrayCoercible) -> Array:
        a, power = ensure_array(a), ensure_array(power)
        ctx.store(a, power)
        return Array(
            a.data**power.data,
            device=a.device,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, power = ctx.saved_arrays
        agrad = power.data * (a.data ** (power.data - 1)) * grad
        return (agrad, None)  # type: ignore


@register(OperatorId.POW, op_requirements=OperatorRequirements.Autograd)
def pow_autograd(a: ArrayCoercible, power: ArrayCoercible) -> Array:
    return Pow.apply(a, ensure_array(power))


@register(OperatorId.RELU)
def relu_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    return Array(
        np.maximum(0, a.data),
        device="cpu_np",
        requires_grad=False,
    )


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        return Array(
            np.maximum(0, a.data),
            device=a.device,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        agrad = np.where(a.data > 0, grad, 0)
        return (agrad,)  # type: ignore


@register(OperatorId.RELU, op_requirements=OperatorRequirements.Autograd)
def relu_autograd(a: ArrayCoercible) -> Array:
    return ReLU.apply(a)

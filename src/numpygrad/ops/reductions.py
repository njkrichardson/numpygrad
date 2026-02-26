import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.registry import register, OperatorRequirements
from numpygrad.core.opid import OperatorId
from numpygrad.core.function import Function, Context
from numpygrad.ops.core import ensure_array
from numpygrad.core.broadcasting import expand_to_shape


@register(OperatorId.SUM)
def sum_cpu(
    a: ArrayCoercible, axis: int | None = None, keepdims: bool = False
) -> Array:
    a = ensure_array(a)
    return Array(
        np.sum(a.data, axis, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


class Sum(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ArrayCoercible, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        a = ensure_array(a)
        ctx.store(a)

        return Array(
            np.sum(a.data, axis, keepdims=keepdims),
            device=a.device,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        return expand_to_shape(grad, a.shape), None, None  # type:ignore


@register(OperatorId.SUM, op_requirements=OperatorRequirements.Autograd)
def sum_autograd(
    a: ArrayCoercible, axis: int | None = None, keepdims: bool = False
) -> Array:
    return Sum.apply(a, axis, keepdims)


@register(OperatorId.MEAN)
def mean_cpu(
    a: ArrayCoercible, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
) -> Array:
    a = ensure_array(a)
    return Array(
        np.mean(a.data, axis, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


class Mean(Function):
    @staticmethod
    def forward(
        ctx: Context,
        a: ArrayCoercible,
        axis: tuple[int, ...] | int | None = None,
        keepdims: bool = False,
    ) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        out = np.mean(a.data, axis, keepdims=keepdims)
        ctx.reduced_shape = out.shape

        return Array(
            out,
            device=a.device,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        reduced_shape = ctx.reduced_shape
        grad_a = grad / np.prod(np.array(reduced_shape))
        return expand_to_shape(grad, a.shape), None, None  # type:ignore


@register(OperatorId.MEAN, op_requirements=OperatorRequirements.Autograd)
def mean_autograd(
    a: ArrayCoercible, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
) -> Array:
    return Mean.apply(a, axis, keepdims)

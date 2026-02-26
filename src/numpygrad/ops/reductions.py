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
            requires_grad=a.requires_grad,
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
    def forward(ctx, a, axis=None, keepdims=False):
        a = ensure_array(a)

        ctx.store(a)
        ctx.axis = axis
        ctx.keepdims = keepdims

        if axis is None:
            ctx.count = a.data.size
        else:
            axis_tuple = axis if isinstance(axis, tuple) else (axis,)
            ctx.count = np.prod([a.shape[i] for i in axis_tuple])

        out = np.mean(a.data, axis=axis, keepdims=keepdims)

        return Array(
            out,
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad):
        a = ctx.saved_arrays[0]

        grad = grad / ctx.count

        grad = expand_to_shape(grad, a.shape)

        return grad, None, None


@register(OperatorId.MEAN, op_requirements=OperatorRequirements.Autograd)
def mean_autograd(
    a: ArrayCoercible, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
) -> Array:
    return Mean.apply(a, axis, keepdims)

import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.registry import register, OperatorRequirements
from numpygrad.core.opid import OperatorId
from numpygrad.core.function import Function, Context
from numpygrad.ops.core import ensure_array


@register(OperatorId.TRANSPOSE)
def mul_cpu(a: ArrayCoercible, axes: tuple[int, ...]) -> Array:
    a = ensure_array(a)
    return Array(
        np.transpose(a.data, axes=axes),
        device="cpu_np",
        requires_grad=False,
    )


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, axes: tuple[int, ...]) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        ctx.axes = axes

        return Array(
            np.transpose(a.data, axes=axes),
            device=a.device,
            requires_grad=True,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        axes = ctx.axes
        inverse_axes = np.argsort(axes)
        return np.transpose(grad, axes=inverse_axes), None


@register(OperatorId.TRANSPOSE, op_requirements=OperatorRequirements.Autograd)
def transpose_autograd(a: ArrayCoercible, axes: tuple[int, ...]) -> Array:
    return Transpose.apply(a, axes)

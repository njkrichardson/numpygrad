import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.registry import register, OperatorRequirements
from numpygrad.core.opid import OperatorId
from numpygrad.core.function import Function, Context
from numpygrad.ops.core import ensure_array


@register(OperatorId.TRANSPOSE)
def tranpose_cpu(a: ArrayCoercible, axes: tuple[int, ...]) -> Array:
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
            requires_grad=a.requires_grad,
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


@register(OperatorId.RESHAPE)
def reshape_cpu(a: ArrayCoercible, new_shape: tuple[int, ...] | int) -> Array:
    a = ensure_array(a)
    return Array(
        np.reshape(a.data, shape=new_shape),
        device="cpu_np",
        requires_grad=False,
    )


class Reshape(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ArrayCoercible, new_shape: tuple[int, ...] | int
    ) -> Array:
        a = ensure_array(a)
        ctx.store(a)

        return Array(
            np.reshape(a.data, shape=new_shape),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        return np.reshape(grad, shape=a.shape), None  # type: ignore


@register(OperatorId.RESHAPE, op_requirements=OperatorRequirements.Autograd)
def reshape_autograd(a: ArrayCoercible, new_shape: tuple[int, ...] | int) -> Array:
    return Reshape.apply(a, new_shape)


def normalize_key(key):
    if isinstance(key, Array):
        return key.data
    elif isinstance(key, tuple):
        return tuple(normalize_key(k) for k in key)
    return key


@register(OperatorId.SLICE)
def slice_cpu(a: ArrayCoercible, key) -> Array:
    a = ensure_array(a)
    key = normalize_key(key)
    return Array(
        a.data[key],
        device="cpu_np",
        requires_grad=False,
    )


class Slice(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, key) -> Array:
        key = normalize_key(key)
        a = ensure_array(a)

        ctx.store(a)
        ctx.key = key

        return Array(
            a.data[key],
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        grad_a = np.zeros_like(a.data, dtype=grad.dtype)
        grad_a[ctx.key] = grad
        return (grad_a, None)


@register(OperatorId.SLICE, op_requirements=OperatorRequirements.Autograd)
def slice_autograd(a: ArrayCoercible, key) -> Array:
    return Slice.apply(a, key)

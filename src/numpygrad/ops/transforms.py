from typing import Any

import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.broadcasting import unbroadcast
from numpygrad.core.function import Context, Function
from numpygrad.core.opid import OperatorId
from numpygrad.core.registry import OperatorRequirements, register
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
        ctx.axes = axes

        return Array(
            np.transpose(a.data, axes=axes),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        assert ctx.axes is not None
        inverse_axes = np.argsort(ctx.axes)
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
    def forward(ctx: Context, a: ArrayCoercible, new_shape: tuple[int, ...] | int) -> Array:
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


def normalize_key(
    key: "ArrayCoercible | slice | tuple[slice, ...]",
) -> Any:
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
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a = ctx.saved_arrays[0]
        grad_a = np.zeros_like(a.data, dtype=grad.dtype)
        grad_a[ctx.key] = grad
        return (grad_a, None)


@register(OperatorId.SLICE, op_requirements=OperatorRequirements.Autograd)
def slice_autograd(a: ArrayCoercible, key) -> Array:
    return Slice.apply(a, key)


@register(OperatorId.UNSQUEEZE)
def unsqueeze_cpu(a: ArrayCoercible, axis: int) -> Array:
    a = ensure_array(a)
    return Array(
        np.expand_dims(a.data, axis=axis),
        device="cpu_np",
        requires_grad=False,
    )


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, axis: int) -> Array:
        a = ensure_array(a)
        ctx.axis = axis

        return Array(
            np.expand_dims(a.data, axis=axis),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        return np.squeeze(grad, axis=ctx.axis), None


@register(OperatorId.UNSQUEEZE, op_requirements=OperatorRequirements.Autograd)
def unsqueeze_autograd(a: ArrayCoercible, axis: int) -> Array:
    return Unsqueeze.apply(a, axis)


@register(OperatorId.FLATTEN)
def flatten_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    return Array(
        np.ravel(a.data),
        device="cpu_np",
        requires_grad=False,
    )


class Flatten(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        ctx.store(a)

        return Array(
            np.ravel(a.data),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a = ctx.saved_arrays[0]
        return np.reshape(grad, shape=a.shape), None


@register(OperatorId.FLATTEN, op_requirements=OperatorRequirements.Autograd)
def flatten_autograd(a: ArrayCoercible) -> Array:
    return Flatten.apply(a)


@register(OperatorId.STACK)
def stack_cpu(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    arrs = tuple(ensure_array(a) for a in arrays)
    return Array(
        np.stack([a.data for a in arrs], axis=axis),
        device="cpu_np",
        requires_grad=any(a.requires_grad for a in arrs),
    )


class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *arrays_and_axis: ArrayCoercible | int) -> Array:
        *array_args, axis = arrays_and_axis
        assert isinstance(axis, int)
        arrays = tuple(ensure_array(a) for a in array_args)
        ctx.axis = axis
        ctx.num_arrays = len(arrays)
        ctx.original_shapes = tuple(a.shape for a in arrays)

        data = [a.data for a in arrays]
        out = np.stack(data, axis=axis)
        return Array(
            out,
            device=arrays[0].device,
            requires_grad=any(a.requires_grad for a in arrays),
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        assert isinstance(ctx.axis, int)
        axis = ctx.axis
        splits = np.split(grad, ctx.num_arrays, axis=axis)
        # Each split has an extra dimension at axis; squeeze it to match input shape.
        return *tuple(np.squeeze(split, axis=axis) for split in splits), None


@register(OperatorId.STACK, op_requirements=OperatorRequirements.Autograd)
def stack_autograd(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    return Stack.apply(*arrays, axis)


@register(OperatorId.CAT)
def cat_cpu(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    arrs = tuple(ensure_array(a) for a in arrays)
    return Array(
        np.concatenate([a.data for a in arrs], axis=axis),
        device="cpu_np",
        requires_grad=any(a.requires_grad for a in arrs),
    )


class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *arrays_and_axis: ArrayCoercible | int) -> Array:
        *array_args, axis = arrays_and_axis
        assert isinstance(axis, int)
        arrays = tuple(ensure_array(a) for a in array_args)
        ctx.axis = axis
        ctx.sizes = tuple(a.shape[axis] for a in arrays)
        ctx.original_shapes = tuple(a.shape for a in arrays)

        data = [a.data for a in arrays]
        out = np.concatenate(data, axis=axis)
        return Array(
            out,
            device=arrays[0].device,
            requires_grad=any(a.requires_grad for a in arrays),
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        assert isinstance(ctx.axis, int)
        axis = ctx.axis
        sizes = ctx.sizes
        splits = np.split(grad, np.cumsum(sizes)[:-1], axis=axis)
        return *tuple(
            unbroadcast(split, shape)
            for split, shape in zip(splits, ctx.original_shapes, strict=False)
        ), None


@register(OperatorId.CAT, op_requirements=OperatorRequirements.Autograd)
def cat_autograd(arrays: tuple[ArrayCoercible, ...], axis: int = 0) -> Array:
    return Cat.apply(*arrays, axis)

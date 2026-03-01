import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.broadcasting import reduce_grad_to_shape
from numpygrad.core.function import Context, Function
from numpygrad.core.opid import OperatorId
from numpygrad.core.registry import OperatorRequirements, register
from numpygrad.ops.core import ensure_array


@register(OperatorId.MATMUL)
def mm_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    a, b = ensure_array(a), ensure_array(b)
    return Array(
        np.matmul(a.data, b.data),
        device="cpu_np",
        requires_grad=False,
    )


class Matmul(Function):
    @staticmethod
    def forward(ctx, a: Array, b: Array) -> Array:
        ctx.store(a, b)
        # save original shapes for backward
        ctx.a_shape = a.data.shape
        ctx.b_shape = b.data.shape

        # handle 1D vectors by temporarily reshaping to 2D
        ctx.squeeze_a = False
        ctx.squeeze_b = False

        a_data = a.data
        b_data = b.data

        if a.ndim == 1:
            a_data = a_data[None, :]  # row vector
            ctx.squeeze_a = True
        if b.ndim == 1:
            b_data = b_data[:, None]  # column vector
            ctx.squeeze_b = True

        # forward matmul (NumPy handles broadcasting in batch dims)
        out_data = np.matmul(a_data, b_data)
        # restore 1D output when we had 1D input (match NumPy/PyTorch semantics)
        if ctx.squeeze_a and out_data.shape[0] == 1:
            out_data = out_data.squeeze(0)
        if ctx.squeeze_b and out_data.shape[-1] == 1:
            out_data = out_data.squeeze(-1)
        return Array(out_data, device=a.device, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_out: np.ndarray):
        a, b = ctx.saved_arrays

        # forward shapes
        a_shape = ctx.a_shape
        b_shape = ctx.b_shape

        a_data = a.data
        b_data = b.data
        grad_data = grad_out

        # reshape 1D inputs to 2D for matmul
        if a_data.ndim == 1:
            a_data = a_data[None, :]  # row vector
            grad_data = grad_data[None, :]
        if b_data.ndim == 1:
            b_data = b_data[:, None]  # column vector
            grad_data = grad_data[:, None] if grad_data.ndim == 1 else grad_data

        # compute raw gradients
        grad_a = np.matmul(grad_data, np.swapaxes(b_data, -1, -2))
        grad_b = np.matmul(np.swapaxes(a_data, -1, -2), grad_data)

        # squeeze back if original inputs were 1D — must happen before
        # reduce_grad_to_shape so the unit dim isn't mistaken for a broadcast axis
        if len(a_shape) == 1 and grad_a.ndim >= 2 and grad_a.shape[0] == 1:
            grad_a = grad_a.squeeze(0)
        if len(b_shape) == 1 and grad_b.ndim >= 2 and grad_b.shape[-1] == 1:
            grad_b = grad_b.squeeze(-1)

        # reduce broadcasted axes
        grad_a = reduce_grad_to_shape(grad_a, a_shape)
        grad_b = reduce_grad_to_shape(grad_b, b_shape)

        return grad_a, grad_b


@register(OperatorId.MATMUL, op_requirements=OperatorRequirements.Autograd)
def mul_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Matmul.apply(a, b)


@register(OperatorId.DOT)
def dot_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    a, b = ensure_array(a), ensure_array(b)
    return Array(
        np.dot(a.data, b.data),
        device="cpu_np",
        requires_grad=False,
    )


class Dot(Function):
    @staticmethod
    def forward(ctx, a: Array, b: Array) -> Array:
        ctx.store(a, b)
        return Array(np.dot(a.data, b.data), device=a.device, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_out: np.ndarray):
        a, b = ctx.saved_arrays
        grad_a = grad_out * b.data
        grad_b = grad_out * a.data
        return grad_a, grad_b


@register(OperatorId.DOT, op_requirements=OperatorRequirements.Autograd)
def dot_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Dot.apply(a, b)


@register(OperatorId.NORM)
def norm_cpu(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    a = ensure_array(a)
    return Array(
        np.linalg.norm(a.data, axis=axis, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


class Norm(Function):
    @staticmethod
    def forward(ctx, a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        out = np.linalg.norm(a.data, axis=axis, keepdims=keepdims)
        ctx.store(a, out)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return Array(out, device=a.device, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_out: np.ndarray):
        a, out = ctx.saved_arrays
        axis = ctx.axis
        keepdims = ctx.keepdims

        # out is the raw ndarray from np.linalg.norm
        out_val = out.data if hasattr(out, "data") else out
        norm_safe = np.where(out_val == 0, 1, out_val)
        if not keepdims and axis is not None:
            grad_out = np.expand_dims(grad_out, axis=axis)
            norm_safe = np.expand_dims(norm_safe, axis=axis)
        grad_a = (grad_out / norm_safe) * a.data
        return grad_a, None, None


@register(OperatorId.NORM, op_requirements=OperatorRequirements.Autograd)
def norm_autograd(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    return Norm.apply(a, axis, keepdims)


@register(OperatorId.DIAGONAL)
def diagonal_cpu(a: ArrayCoercible, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Array:
    a = ensure_array(a)
    return Array(
        np.diagonal(a.data, offset=offset, axis1=axis1, axis2=axis2),
        device="cpu_np",
        requires_grad=False,
    )


class Diagonal(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ArrayCoercible, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> Array:
        a = ensure_array(a)
        ctx.input_shape = a.shape
        ctx.offset = offset
        ctx.axis1 = axis1
        ctx.axis2 = axis2
        # np.diagonal returns a read-only view; copy to make it writable
        out = np.diagonal(a.data, offset=offset, axis1=axis1, axis2=axis2).copy()
        return Array(out, device=a.device, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        input_shape = ctx.input_shape
        offset = ctx.offset
        axis1 = ctx.axis1
        axis2 = ctx.axis2

        # Move the two axes of interest to positions 0 and 1
        other_axes = [i for i in range(len(input_shape)) if i != axis1 and i != axis2]
        perm = [axis1, axis2] + other_axes
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i

        permuted_shape = tuple(input_shape[i] for i in perm)
        grad_permuted = np.zeros(permuted_shape, dtype=grad.dtype)

        diag_len = grad.shape[-1] if grad.ndim > 1 else grad.shape[0]
        # For N-D arrays, grad has shape (...batch..., diag_len); move diag_len to end
        # grad after np.diagonal has shape (*batch_dims, diag_len) where batch_dims
        # are the non-axis1/axis2 dims in original order — match with other_axes order.
        if offset >= 0:
            rows = np.arange(diag_len)
            cols = np.arange(offset, offset + diag_len)
        else:
            rows = np.arange(-offset, -offset + diag_len)
            cols = np.arange(diag_len)

        # grad has shape (*other_axes_sizes, diag_len); index into permuted grad
        if len(other_axes) == 0:
            grad_permuted[rows, cols] = grad
        else:
            # grad shape: (*other_dims, diag_len); we need to scatter into [rows, cols, ...]
            # Use advanced indexing: grad_permuted[rows, cols, ...] += grad[..., :]
            # Transpose grad to (diag_len, *other_dims) for broadcasting
            grad_t = np.moveaxis(grad, -1, 0)  # (diag_len, *other_dims)
            grad_permuted[rows, cols] = grad_t

        grad_out = np.transpose(grad_permuted, inv_perm)
        return grad_out, None, None, None


@register(OperatorId.DIAGONAL, op_requirements=OperatorRequirements.Autograd)
def diagonal_autograd(a: ArrayCoercible, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Array:
    return Diagonal.apply(a, offset, axis1, axis2)

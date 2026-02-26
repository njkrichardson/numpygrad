import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.registry import register, OperatorRequirements
from numpygrad.core.opid import OperatorId
from numpygrad.core.function import Function, Context
from numpygrad.ops.core import ensure_array
from numpygrad.core.broadcasting import reduce_grad_to_shape


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
        ctx.a = a
        ctx.b = b
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
        return Array(out_data, device=a.device, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_out: np.ndarray):
        a = ctx.a
        b = ctx.b

        # forward shapes
        a_shape = ctx.a_shape
        b_shape = ctx.b_shape

        # make copies for computation
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

        # reduce broadcasted axes
        grad_a = reduce_grad_to_shape(grad_a, a_shape)
        grad_b = reduce_grad_to_shape(grad_b, b_shape)

        # squeeze back if original inputs were 1D (only if we still have the added dim)
        if len(a_shape) == 1 and grad_a.ndim >= 2 and grad_a.shape[0] == 1:
            grad_a = grad_a.squeeze(0)
        if len(b_shape) == 1 and grad_b.ndim >= 2 and grad_b.shape[-1] == 1:
            grad_b = grad_b.squeeze(-1)

        return grad_a, grad_b


@register(OperatorId.MATMUL, op_requirements=OperatorRequirements.Autograd)
def mul_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Matmul.apply(a, b)

import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.function import Context, Function
from numpygrad.core.opid import OperatorId
from numpygrad.core.registry import OperatorRequirements, register
from numpygrad.ops.core import ArrayCoercible, ensure_array


@register(OperatorId.SOFTMAX)
def softmax_cpu(a: ArrayCoercible, axis: int = -1) -> Array:
    a = ensure_array(a)
    x = a.data - np.max(a.data, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return Array(
        exp_x / np.sum(exp_x, axis=axis, keepdims=True), device="cpu_np", requires_grad=False
    )


class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, axis: int = -1) -> Array:
        a = ensure_array(a)
        x = a.data - np.max(a.data, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        s = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        out = Array(s, device=a.device, requires_grad=a.requires_grad)
        ctx.store(a, out)
        ctx.axis = axis
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        _, out = ctx.saved_arrays
        s = out.data if isinstance(out, Array) else out
        dotprod = np.sum(s * grad, axis=ctx.axis, keepdims=True)
        return s * (grad - dotprod), None  # type: ignore[return-value]


@register(OperatorId.SOFTMAX, op_requirements=OperatorRequirements.Autograd)
def softmax_autograd(a: ArrayCoercible, axis: int = -1) -> Array:
    return Softmax.apply(a, axis)


@register(OperatorId.LOG_SOFTMAX)
def log_softmax_cpu(a: ArrayCoercible, axis: int = -1) -> Array:
    a = ensure_array(a)
    x = a.data - np.max(a.data, axis=axis, keepdims=True)
    lsm = x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))
    return Array(lsm, device="cpu_np", requires_grad=False)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, axis: int = -1) -> Array:
        a = ensure_array(a)
        x = a.data - np.max(a.data, axis=axis, keepdims=True)
        lsm = x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))
        out = Array(lsm, device=a.device, requires_grad=a.requires_grad)
        ctx.store(a, out)
        ctx.axis = axis
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        _, out = ctx.saved_arrays
        lsm = out.data if isinstance(out, Array) else out
        s = np.exp(lsm)
        return grad - s * np.sum(grad, axis=ctx.axis, keepdims=True), None  # type: ignore[return-value]


@register(OperatorId.LOG_SOFTMAX, op_requirements=OperatorRequirements.Autograd)
def log_softmax_autograd(a: ArrayCoercible, axis: int = -1) -> Array:
    return LogSoftmax.apply(a, axis)

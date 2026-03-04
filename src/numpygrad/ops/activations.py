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
        ctx.store(s)
        ctx.axis = axis
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        (s,) = ctx.saved_arrays
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
        ctx.store(lsm)
        ctx.axis = axis
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        (lsm,) = ctx.saved_arrays
        s = np.exp(lsm)
        return grad - s * np.sum(grad, axis=ctx.axis, keepdims=True), None  # type: ignore[return-value]


@register(OperatorId.LOG_SOFTMAX, op_requirements=OperatorRequirements.Autograd)
def log_softmax_autograd(a: ArrayCoercible, axis: int = -1) -> Array:
    return LogSoftmax.apply(a, axis)


@register(OperatorId.SIGMOID)
def sigmoid_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    s = 1.0 / (1.0 + np.exp(-a.data))
    return Array(s, device="cpu_np", requires_grad=False)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        s = 1.0 / (1.0 + np.exp(-a.data))
        out = Array(s, device=a.device, requires_grad=a.requires_grad)
        ctx.store(s)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        (s,) = ctx.saved_arrays
        return (grad * s * (1.0 - s),)


@register(OperatorId.SIGMOID, op_requirements=OperatorRequirements.Autograd)
def sigmoid_autograd(a: ArrayCoercible) -> Array:
    return Sigmoid.apply(a)


@register(OperatorId.TANH)
def tanh_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    t = np.tanh(a.data)
    return Array(t, device="cpu_np", requires_grad=False)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        t = np.tanh(a.data)
        out = Array(t, device=a.device, requires_grad=a.requires_grad)
        ctx.store(t)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        (t,) = ctx.saved_arrays
        return (grad * (1.0 - t**2),)


@register(OperatorId.TANH, op_requirements=OperatorRequirements.Autograd)
def tanh_autograd(a: ArrayCoercible) -> Array:
    return Tanh.apply(a)


@register(OperatorId.SOFTPLUS)
def softplus_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    return Array(np.logaddexp(0, a.data), device="cpu_np", requires_grad=False)


class SoftPlus(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        out = Array(np.logaddexp(0, a.data), device=a.device, requires_grad=a.requires_grad)
        ctx.store(a)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        (a,) = ctx.saved_arrays
        return (grad / (1.0 + np.exp(-a.data)),)


@register(OperatorId.SOFTPLUS, op_requirements=OperatorRequirements.Autograd)
def softplus_autograd(a: ArrayCoercible) -> Array:
    return SoftPlus.apply(a)


_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_GELU_COEFF = 0.044715


@register(OperatorId.GELU)
def gelu_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    x = a.data
    out = 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + _GELU_COEFF * x**3)))
    return Array(out, device=a.device, requires_grad=False)


class GELU(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        x = a.data
        out = 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + _GELU_COEFF * x**3)))
        return Array(out, device=a.device, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        (a,) = ctx.saved_arrays
        x = a.data
        inner = _SQRT_2_OVER_PI * (x + _GELU_COEFF * x**3)
        t = np.tanh(inner)
        du_dx = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_COEFF * x**2)
        d_gelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du_dx
        return (grad * d_gelu,)


@register(OperatorId.GELU, op_requirements=OperatorRequirements.Autograd)
def gelu_autograd(a: ArrayCoercible) -> Array:
    return GELU.apply(a)

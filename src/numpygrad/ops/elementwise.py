import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.broadcasting import unbroadcast
from numpygrad.core.function import Context, Function
from numpygrad.core.opid import OperatorId
from numpygrad.core.registry import OperatorRequirements, register
from numpygrad.ops.core import ensure_array


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
            requires_grad=a.requires_grad or b.requires_grad,
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
            requires_grad=a.requires_grad or b.requires_grad,
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
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, power = ctx.saved_arrays
        # agrad = power.data * (a.data ** (power.data - 1)) * grad
        # return (agrad, None)  # type: ignore
        agrad = power.data * (a.data ** (power.data - 1)) * grad
        agrad = unbroadcast(agrad, a.shape)
        return agrad, None


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
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        agrad = np.where(a.data > 0, grad, 0)
        return (agrad,)  # type: ignore


@register(OperatorId.RELU, op_requirements=OperatorRequirements.Autograd)
def relu_autograd(a: ArrayCoercible) -> Array:
    return ReLU.apply(a)


@register(OperatorId.EXP)
def exp_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    return Array(
        np.exp(a.data),
        device="cpu_np",
        requires_grad=False,
    )


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        return Array(
            np.exp(a.data),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        agrad = grad * np.exp(a.data)
        return (agrad,)


@register(OperatorId.EXP, op_requirements=OperatorRequirements.Autograd)
def exp_autograd(a: ArrayCoercible) -> Array:
    return Exp.apply(a)


@register(OperatorId.LOG)
def log_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    return Array(
        np.log(a.data),
        device="cpu_np",
        requires_grad=False,
    )


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        return Array(
            np.log(a.data),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        agrad = grad * 1 / a.data
        return (agrad,)


@register(OperatorId.LOG, op_requirements=OperatorRequirements.Autograd)
def log_autograd(a: ArrayCoercible) -> Array:
    return Log.apply(a)


@register(OperatorId.ABS)
def abs_cpu(a: ArrayCoercible) -> Array:
    a = ensure_array(a)
    return Array(
        np.abs(a.data),
        device="cpu_np",
        requires_grad=False,
    )


class Abs(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        return Array(
            np.abs(a.data),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        agrad = grad * np.sign(a.data)
        return (agrad,)


@register(OperatorId.ABS, op_requirements=OperatorRequirements.Autograd)
def abs_autograd(a: ArrayCoercible) -> Array:
    return Abs.apply(a)


@register(OperatorId.CLIP)
def clip_cpu(a: ArrayCoercible, min: ArrayCoercible, max: ArrayCoercible) -> Array:
    a, min, max = ensure_array(a), ensure_array(min), ensure_array(max)
    return Array(
        np.clip(a.data, min.data, max.data),
        device="cpu_np",
        requires_grad=False,
    )


class Clip(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, min: ArrayCoercible, max: ArrayCoercible) -> Array:
        a, min, max = ensure_array(a), ensure_array(min), ensure_array(max)
        ctx.store(a, min, max)
        return Array(
            np.clip(a.data, min.data, max.data),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, min, max = ctx.saved_arrays
        agrad = np.where(a.data < min.data, 0, 1) * np.where(a.data > max.data, 0, 1) * grad
        return (agrad, None, None)


@register(OperatorId.CLIP, op_requirements=OperatorRequirements.Autograd)
def clip_autograd(a: ArrayCoercible, min: ArrayCoercible, max: ArrayCoercible) -> Array:
    return Clip.apply(a, min, max)


@register(OperatorId.MAXIMUM)
def maximum_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    a, b = ensure_array(a), ensure_array(b)
    return Array(
        np.maximum(a.data, b.data),
        device="cpu_np",
        requires_grad=False,
    )


class Maximum(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, b: ArrayCoercible) -> Array:
        a, b = ensure_array(a), ensure_array(b)
        ctx.store(a, b)
        return Array(
            np.maximum(a.data, b.data),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = ctx.saved_arrays
        amask = a.data > b.data
        bmask = b.data > a.data
        equal_mask = a.data == b.data

        # split tied grads equally (torch behavior)
        grad_a = grad * (amask + 0.5 * equal_mask)
        grad_b = grad * (bmask + 0.5 * equal_mask)

        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)

        return (grad_a, grad_b)


@register(OperatorId.MAXIMUM, op_requirements=OperatorRequirements.Autograd)
def maximum_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Maximum.apply(a, b)


@register(OperatorId.MINIMUM)
def minimum_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    a, b = ensure_array(a), ensure_array(b)
    return Array(
        np.minimum(a.data, b.data),
        device="cpu_np",
        requires_grad=False,
    )


class Minimum(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, b: ArrayCoercible) -> Array:
        a, b = ensure_array(a), ensure_array(b)
        ctx.store(a, b)
        return Array(
            np.minimum(a.data, b.data),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = ctx.saved_arrays
        amask = a.data < b.data
        bmask = b.data < a.data
        equal_mask = a.data == b.data

        # split tied grads equally (torch behavior)
        grad_a = grad * (amask + 0.5 * equal_mask)
        grad_b = grad * (bmask + 0.5 * equal_mask)

        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)

        return (grad_a, grad_b)


@register(OperatorId.MINIMUM, op_requirements=OperatorRequirements.Autograd)
def minimum_autograd(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Minimum.apply(a, b)

import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.broadcasting import expand_to_shape
from numpygrad.core.function import Context, Function
from numpygrad.core.opid import OperatorId
from numpygrad.core.registry import OperatorRequirements, register
from numpygrad.ops.core import ensure_array


@register(OperatorId.SUM)
def sum_cpu(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
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
        ctx.axis = axis
        ctx.keepdims = keepdims

        return Array(
            np.sum(a.data, axis, keepdims=keepdims),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        axis = ctx.axis
        keepdims = ctx.keepdims
        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis=axis)
        return expand_to_shape(grad, a.shape), None, None  # type:ignore


@register(OperatorId.SUM, op_requirements=OperatorRequirements.Autograd)
def sum_autograd(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
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
        axis = ctx.axis
        keepdims = ctx.keepdims

        grad = grad / ctx.count
        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis=axis)
        grad = expand_to_shape(grad, a.shape)

        return grad, None, None


@register(OperatorId.MEAN, op_requirements=OperatorRequirements.Autograd)
def mean_autograd(
    a: ArrayCoercible, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
) -> Array:
    return Mean.apply(a, axis, keepdims)


@register(OperatorId.MAX)
def max_cpu(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    a = ensure_array(a)
    return Array(
        np.max(a.data, axis, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


class Max(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ArrayCoercible, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        ctx.axis = axis
        ctx.keepdims = keepdims

        return Array(
            np.max(a.data, axis, keepdims=keepdims),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        axis = ctx.axis
        keepdims = ctx.keepdims

        # Use keepdims=True so amax/count broadcast with a.data
        amax = np.max(a.data, axis, keepdims=True)
        amask = a.data == amax
        count = np.sum(amask, axis, keepdims=True)

        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis=axis)

        grad = grad / count
        grad_a = grad * amask
        return grad_a, None, None  # type:ignore


@register(OperatorId.MAX, op_requirements=OperatorRequirements.Autograd)
def max_autograd(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    return Max.apply(a, axis, keepdims)


@register(OperatorId.MIN)
def min_cpu(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    a = ensure_array(a)
    return Array(
        np.min(a.data, axis, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


class Min(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ArrayCoercible, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        ctx.axis = axis
        ctx.keepdims = keepdims

        return Array(
            np.min(a.data, axis, keepdims=keepdims),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        a = ctx.saved_arrays[0]
        axis = ctx.axis
        keepdims = ctx.keepdims

        # Use keepdims=True so amin/count broadcast with a.data
        amin = np.min(a.data, axis, keepdims=True)
        amask = a.data == amin
        count = np.sum(amask, axis, keepdims=True)

        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis=axis)

        grad = grad / count
        grad_a = grad * amask
        return grad_a, None, None  # type:ignore


@register(OperatorId.MIN, op_requirements=OperatorRequirements.Autograd)
def min_autograd(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    return Min.apply(a, axis, keepdims)


@register(OperatorId.PRODUCT)
def product_cpu(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    a = ensure_array(a)
    return Array(
        np.prod(a.data, axis, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


class Product(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ArrayCoercible, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        a = ensure_array(a)
        out = np.prod(a.data, axis, keepdims=keepdims)
        ctx.store(a, out)
        ctx.axis = axis
        ctx.keepdims = keepdims

        return Array(
            out,
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, out = ctx.saved_arrays
        axis = ctx.axis
        keepdims = ctx.keepdims

        # a is Array; out is the raw ndarray from np.prod (stored as second element)
        out_nd = out.data if isinstance(out, Array) else out
        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis)
            out_nd = np.expand_dims(out_nd, axis)

        grad_a = grad * (out_nd / a.data)
        return grad_a, None, None


@register(OperatorId.PRODUCT, op_requirements=OperatorRequirements.Autograd)
def product_autograd(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    return Product.apply(a, axis, keepdims)


@register(OperatorId.ARGMAX)
def argmax_cpu(a: ArrayCoercible, axis: int | None = None, keepdims: bool = False) -> Array:
    a = ensure_array(a)
    return Array(
        np.argmax(a.data, axis, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


@register(OperatorId.VAR)
def var_cpu(
    a: ArrayCoercible, axis: int | None = None, ddof: int = 0, keepdims: bool = False
) -> Array:
    a = ensure_array(a)
    return Array(
        np.var(a.data, axis=axis, ddof=ddof, keepdims=keepdims),
        device="cpu_np",
        requires_grad=False,
    )


class Var(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ArrayCoercible, axis=None, ddof: int = 0, keepdims: bool = False
    ) -> Array:
        a = ensure_array(a)
        ctx.store(a)
        ctx.axis = axis
        ctx.ddof = ddof
        ctx.keepdims = keepdims
        return Array(
            np.var(a.data, axis=axis, ddof=ddof, keepdims=keepdims),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (a,) = ctx.saved_arrays
        axis = ctx.axis
        ddof = ctx.ddof
        keepdims = ctx.keepdims

        if axis is None:
            n = a.data.size
        else:
            axis_tuple = axis if isinstance(axis, tuple) else (axis,)
            n = int(np.prod([a.shape[i] for i in axis_tuple]))

        mean_x = np.mean(a.data, axis=axis, keepdims=True)
        upstream = grad if keepdims or axis is None else np.expand_dims(grad, axis=axis)
        grad_a = upstream * 2.0 * (a.data - mean_x) / (n - ddof)
        return grad_a, None, None, None


@register(OperatorId.VAR, op_requirements=OperatorRequirements.Autograd)
def var_autograd(a: ArrayCoercible, axis=None, ddof: int = 0, keepdims: bool = False) -> Array:
    return Var.apply(a, axis, ddof, keepdims)


@register(OperatorId.CUMSUM)
def cumsum_cpu(a: ArrayCoercible, axis=None) -> Array:
    a = ensure_array(a)
    return Array(
        np.cumsum(a.data, axis=axis),
        device="cpu_np",
        requires_grad=False,
    )


class CumSum(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, axis=None) -> Array:
        a = ensure_array(a)
        ctx.input_shape = a.shape
        ctx.axis = axis
        return Array(
            np.cumsum(a.data, axis=axis),
            device=a.device,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        axis = ctx.axis
        if axis is None:
            flat_grad = np.flip(np.cumsum(np.flip(grad)))
            return flat_grad.reshape(ctx.input_shape), None
        else:
            return np.flip(np.cumsum(np.flip(grad, axis=axis), axis=axis), axis=axis), None


@register(OperatorId.CUMSUM, op_requirements=OperatorRequirements.Autograd)
def cumsum_autograd(a: ArrayCoercible, axis=None) -> Array:
    return CumSum.apply(a, axis)


@register(OperatorId.CUMPROD)
def cumprod_cpu(a: ArrayCoercible, axis=None) -> Array:
    a = ensure_array(a)
    return Array(
        np.cumprod(a.data, axis=axis),
        device="cpu_np",
        requires_grad=False,
    )


class CumProd(Function):
    @staticmethod
    def forward(ctx: Context, a: ArrayCoercible, axis=None) -> Array:
        a = ensure_array(a)
        out = np.cumprod(a.data, axis=axis)
        ctx.store(a, out)
        ctx.input_shape = a.shape
        ctx.axis = axis
        return Array(out, device=a.device, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, out = ctx.saved_arrays
        out_data = out.data if hasattr(out, "data") else out
        axis = ctx.axis

        # reverse cumsum of (grad * y), then divide by x
        # grad_x[i] = sum_{j>=i} grad[j] * prod_{k in [i+1..j]} x[k]
        #           = (1/x[i]) * reverse_cumsum(grad * y)[i]
        product = grad * out_data
        if axis is None:
            rev_cs = np.flip(np.cumsum(np.flip(product)))
            rev_cs = rev_cs.reshape(ctx.input_shape)
            a_data = a.data
        else:
            rev_cs = np.flip(np.cumsum(np.flip(product, axis=axis), axis=axis), axis=axis)
            a_data = a.data

        grad_a = rev_cs / a_data
        return grad_a, None


@register(OperatorId.CUMPROD, op_requirements=OperatorRequirements.Autograd)
def cumprod_autograd(a: ArrayCoercible, axis=None) -> Array:
    return CumProd.apply(a, axis)

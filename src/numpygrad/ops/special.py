import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.registry import register, OperatorRequirements
from numpygrad.core.opid import OperatorId
from numpygrad.ops.core import ensure_array
from numpygrad.ops.transforms import normalize_key
from numpygrad.core.function import Function


@register(OperatorId.GT)
def gt_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Array(ensure_array(a).data > ensure_array(b).data)


@register(OperatorId.LT)
def lt_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Array(ensure_array(a).data < ensure_array(b).data)


@register(OperatorId.GE)
def ge_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Array(ensure_array(a).data >= ensure_array(b).data)


@register(OperatorId.LE)
def le_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Array(ensure_array(a).data <= ensure_array(b).data)


@register(OperatorId.EQ)
def eq_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Array(ensure_array(a).data == ensure_array(b).data)


@register(OperatorId.NE)
def ne_cpu(a: ArrayCoercible, b: ArrayCoercible) -> Array:
    return Array(ensure_array(a).data != ensure_array(b).data)



@register(OperatorId.SETITEM)
def setitem_cpu(a: ArrayCoercible, key: tuple[int, ...], value: ArrayCoercible) -> Array:
    a, value = ensure_array(a), ensure_array(value)
    key = normalize_key(key)

    out = a.data.copy()
    out[key] = value.data
    return Array(out, device=a.device, requires_grad=a.requires_grad or value.requires_grad)

class Setitem(Function):
    @staticmethod
    def forward(ctx, a: Array, key: tuple[int, ...], value: Array) -> Array:
        a, value = ensure_array(a), ensure_array(value)
        key = normalize_key(key)

        ctx.a_shape = a.shape
        ctx.key = key

        out = a.data.copy()
        out[key] = value.data
        return Array(
            out,
            device=a.device,
            requires_grad=a.requires_grad or value.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_out: np.ndarray):
        key = ctx.key

        grad_a = grad_out.copy()
        grad_a[key] = 0

        grad_value = grad_out[key]

        return grad_a, None, grad_value


@register(OperatorId.SETITEM, op_requirements=OperatorRequirements.Autograd)
def setitem_autograd(a: ArrayCoercible, key: tuple[int, ...], value: ArrayCoercible) -> Array:
    return Setitem.apply(a, key, value)

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.registry import register
from numpygrad.core.opid import OperatorId
from numpygrad.ops.core import ensure_array


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

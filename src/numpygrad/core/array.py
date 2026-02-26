from typing import TypeAlias

import numpy as np

from numpygrad.core.dispatch import dispatch
from numpygrad.core.opid import OperatorId
from numpygrad.core.device import DeviceId

ArrayCoercible: TypeAlias = (
    "np.ndarray | int | float | list[int | float] | tuple[int | float] | Array"
)


class Array:
    def __init__(
        self,
        data: ArrayCoercible,
        *,
        device: DeviceId | str = "cpu_np",
        requires_grad: bool = False,
        label: str = "",
    ):
        if isinstance(data, (int, float)):
            data = np.array(data)
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            pass

        self.data: np.ndarray = data  # type: ignore
        self.device: DeviceId = DeviceId(device)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None

        self.grad_fn = None
        self.ctx = None
        self.parents = ()
        self.label = label

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        out = f"Array(data={self.data}"
        if self.requires_grad:
            out += ", requires_grad=True"
            if self.grad_fn is not None:
                out += f", grad_fn={self.grad_fn.__name__}"
        if self.label != "":
            out += f", label={self.label}"
        out += ")"
        return out

    def __mul__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.MUL, self, other)

    def __add__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.ADD, self, other)

    def __rmul__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.MUL, self, other)

    def __neg__(self) -> "Array":
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.POW, self, other)

    def __truediv__(self, other: "int | float | Array | np.ndarray") -> "Array":
        return self * other**-1

    def __matmul__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.MATMUL, self, other)

    def sum(self, axis=None, keepdims=False) -> "Array":
        return dispatch(OperatorId.SUM, self, axis=axis, keepdims=keepdims)

    def transpose(self, axes: tuple[int, ...]) -> "Array":
        return dispatch(OperatorId.TRANSPOSE, self, axes=axes)

    def reshape(self, new_shape: tuple[int, ...] | int) -> "Array":
        return dispatch(OperatorId.RESHAPE, self, new_shape=new_shape)

    def view(self, new_shape: tuple[int, ...] | int) -> "Array":
        return self.reshape(new_shape)

    def mean(
        self, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
    ) -> "Array":
        return dispatch(OperatorId.MEAN, self, axis=axis, keepdims=keepdims)

    @property
    def T(self) -> "Array":
        return self.transpose(axes=tuple(reversed(range(self.ndim))))

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            grad = np.ones_like(self.data)

        # build topo order
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in getattr(node, "parents", ()):
                    build(parent)
                topo.append(node)

        build(self)

        # seed gradient
        self.grad = grad

        # backward pass
        for node in reversed(topo):
            if getattr(node, "grad_fn", None) is None:
                continue

            grads = node.grad_fn.backward(node.ctx, node.grad)

            for parent, parent_grad in zip(node.parents, grads):
                if getattr(parent, "requires_grad", False):
                    parent.grad += parent_grad

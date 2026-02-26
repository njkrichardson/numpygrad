from typing import Any, TypeAlias, overload

import numpy as np

from numpygrad.core.dispatch import dispatch
from numpygrad.core.opid import OperatorId
from numpygrad.core.device import DeviceId

ArrayConsumable: TypeAlias = (
    np.ndarray | int | float | list[int | float] | tuple[int | float]
)


def unbroadcast(grad, original_shape):
    ndim_added = grad.ndim - len(original_shape)
    for _ in range(ndim_added):
        grad = grad.sum(axis=0)

    for i, (og, _) in enumerate(zip(original_shape, grad.shape)):
        if og == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Array:
    def __init__(
        self,
        data: ArrayConsumable,
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

        self.data = data
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

    def __mul__(self, other: "Array") -> "Array":
        return dispatch(OperatorId.MUL, self, other)

    def __add__(self, other: "Array") -> "Array":
        return dispatch(OperatorId.ADD, self, other)

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            grad = np.ones_like(self.data)

        # build topo order
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build(parent)
                topo.append(node)

        build(self)

        # seed gradient
        self.grad = grad

        # backward pass
        for node in reversed(topo):
            if node.grad_fn is None:
                continue

            grads = node.grad_fn.backward(node.ctx, node.grad)

            for parent, parent_grad in zip(node.parents, grads):
                if parent.requires_grad:
                    parent.grad += parent_grad

    # def _array(self, other: Any) -> "Array":
    #     if isinstance(other, Array):
    #         return other
    #     elif isinstance(other, np.ndarray):
    #         return Array(other)
    #     elif isinstance(other, (int, float)):
    #         return Array(other)
    #     else:
    #         raise NotImplementedError

    # def __add__(self, _other: "ArrayConsumable | Array") -> "Array":
    #     other = self._array(_other)
    #     out = Array(self.data + other.data, (self, other), Operation.ADD)
    #
    #     def vjp():
    #         self.grad += unbroadcast(out.grad, self.data.shape)
    #         other.grad += unbroadcast(out.grad, other.data.shape)
    #
    #     out.vjp = vjp
    #     return out
    #
    # def __neg__(self) -> "Array":
    #     return self * np.array(-1)
    #
    # def __sub__(self, other):
    #     return self + (-other)
    #
    # def __pow__(self, other: int | float) -> "Array":
    #     assert isinstance(other, (int, float))
    #     if other == -1:
    #         label = "/"
    #         op = Operation.DIV
    #     else:
    #         label = f"**{other}"
    #         op = Operation.POW
    #
    #     out = Array(self.data**other, (self,), label=label, op=op)
    #
    #     def vjp():
    #         self.grad += other * (self.data ** (other - 1)) * out.grad
    #
    #     out.vjp = vjp
    #     return out
    #
    # def __truediv__(self, _other: ArrayConsumable) -> "Array":
    #     other = self._array(_other)
    #     return self * other**-1
    #
    # def __mul__(self, _other: ArrayConsumable) -> "Array":
    #     other = self._array(_other)
    #     out = Array(self.data * other.data, (self, other), Operation.MUL)
    #
    #     def vjp():
    #         self.grad += other.data * out.grad
    #         other.grad += self.data * out.grad
    #
    #     out.vjp = vjp
    #     return out
    #
    # def __rmul__(self, other: "Array | np.ndarray") -> "Array":
    #     return self * other
    #
    # def relu(self) -> "Array":
    #     out = Array(np.maximum(0, self.data), (self,), Operation.RELU)
    #
    #     def vjp():
    #         self.grad += (self.data > 0) * out.grad
    #
    #     out.vjp = vjp
    #     return out

    # def backward(self) -> None:
    #     def toposort(node: Array) -> list[Array]:
    #         sorted_nodes = []
    #         visited: set[Array] = set()
    #
    #         def _toposort(_node: Array):
    #             if _node not in visited:
    #                 visited.add(_node)
    #                 for child in _node.children:
    #                     _toposort(child)
    #                 sorted_nodes.append(_node)
    #
    #         _toposort(node)
    #         return sorted_nodes
    #
    #     self.grad = np.ones_like(self.data)
    #     topo = toposort(self)
    #     for node in reversed(topo):
    #         node.vjp()

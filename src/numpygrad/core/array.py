from typing import Any, TypeAlias

import numpy as np

from numpygrad.core.ops import Operation

ArrayConsumable: TypeAlias = (
    "np.ndarray | int | float | list[int | float] | tuple[int | float]"
)


class Array:
    def __init__(
        self,
        data: ArrayConsumable,
        children: tuple["Array"] | tuple = (),
        op: Operation | None = None,
        label: str = "",
    ):
        if isinstance(data, (int, float)):
            data = np.array(data)
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            pass

        self.data = data
        self.children: set["Array"] = set(children)
        self.op = op
        self.label = label
        self.grad = np.zeros(1)
        self.vjp = lambda: None

    def __repr__(self) -> str:
        return f"Array(data={self.data})"

    def _array(self, other: Any) -> "Array":
        if isinstance(other, Array):
            return other
        elif isinstance(other, np.ndarray):
            return Array(other)
        elif isinstance(other, (int, float)):
            return Array(other)
        else:
            raise NotImplementedError

    def __add__(self, _other: ArrayConsumable) -> "Array":
        other = self._array(_other)
        out = Array(self.data + other.data, (self, other), Operation.ADD)

        def vjp():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out.vjp = vjp
        return out

    def __neg__(self) -> "Array":
        return self * np.array(-1)

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other: int | float) -> "Array":
        assert isinstance(other, (int, float))
        if other == -1:
            label = "/"
            op = Operation.DIV
        else:
            label = f"**{other}"
            op = Operation.POW

        out = Array(self.data**other, (self,), label=label, op=op)

        def vjp():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out.vjp = vjp
        return out

    def __truediv__(self, _other: ArrayConsumable) -> "Array":
        other = self._array(_other)
        return self * other**-1

    def __mul__(self, _other: ArrayConsumable) -> "Array":
        other = self._array(_other)
        out = Array(self.data * other.data, (self, other), Operation.MUL)

        def vjp():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.vjp = vjp
        return out

    def __rmul__(self, other: "Array | np.ndarray") -> "Array":
        return self * other

    def relu(self) -> "Array":
        out = Array(np.maximum(0, self.data), (self,), Operation.RELU)

        def vjp():
            self.grad += (self.data > 0) * out.grad

        out.vjp = vjp
        return out

    def backward(self) -> None:
        def toposort(node: Array) -> list[Array]:
            sorted_nodes = []
            visited: set[Array] = set()

            def _toposort(_node: Array):
                if _node not in visited:
                    visited.add(_node)
                    for child in _node.children:
                        _toposort(child)
                    sorted_nodes.append(_node)

            _toposort(node)
            return sorted_nodes

        self.grad = np.ones(1)
        topo = toposort(self)
        for node in reversed(topo):
            node.vjp()

import enum
from pathlib import Path
from typing import Any

import graphviz
import numpy as np


class Operation(enum.StrEnum):
    ADD = "+"
    MUL = "*"
    POW = "**"
    DIV = "/"


class Array:
    def __init__(
        self,
        data: np.ndarray,
        children: tuple["Array"] | tuple = (),
        op: Operation | None = None,
        label: str = "",
    ):
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
        else:
            raise NotImplementedError

    def __add__(self, other: "Array | np.ndarray") -> "Array":
        other = self._array(other)
        out = Array(self.data + other.data, (self, other), Operation.ADD)

        def vjp():
            print("called add.vjp()")
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

    def __truediv__(self, other: "Array | np.ndarray") -> "Array":
        return self * other**-1

    def __mul__(self, other: "Array | np.ndarray") -> "Array":
        other = self._array(other)
        out = Array(self.data * other.data, (self, other), Operation.MUL)

        def vjp():
            print("called mul.vjp()")
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.vjp = vjp
        return out

    def __rmul__(self, other: "Array | np.ndarray") -> "Array":
        return self * other

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


def trace(root: Array) -> tuple[set[Array], set[tuple[Array, Array]]]:
    nodes, edges = set(), set()

    def construct(node: Array):
        if node not in nodes:
            nodes.add(node)
            for child in node.children:
                edges.add((child, node))
                construct(child)

    construct(root)
    return nodes, edges


def draw_computation_graph(root: Array, save_path: Path | None = None) -> None:
    dot = graphviz.Digraph(format="png", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        print(node.label)
        dot.node(
            name=uid,
            label=f"{{{node.label} | {node.data} | {node.grad}}}",
            shape="record",
        )
        if node.op:
            dot.node(name=uid + str(node.op), label=node.op)
            dot.edge(uid + node.op, uid)

    for n1, n2 in edges:
        assert n2.op is not None
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    dot.render(
        filename="graph" if not save_path else str(save_path),
        format="png",
        cleanup=True,
    )


def main():
    a = Array(np.array(2.0), label="a")
    b = Array(np.array(-3.0), label="b")
    c = Array(np.array(10.0), label="c")
    e = a * b
    e.label = "e"
    d = e / c
    d.label = "d"
    f = Array(np.array(-2.0), label="f")
    L = d * f
    L.label = "L"
    L.backward()
    draw_computation_graph(L)


if __name__ == "__main__":
    main()

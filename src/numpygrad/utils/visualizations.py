from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import graphviz

from numpygrad.configuration import MEDIA_DIR
from numpygrad.utils.io import now
from numpygrad.utils.logging import CustomLogger

if TYPE_CHECKING:
    from numpygrad.core.array import Array

Log = CustomLogger(__name__)


def trace(
    root: Array,
) -> tuple[set[Array], set[tuple[Array, Array]]]:
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
    dot = graphviz.Digraph(
        format="png", graph_attr={"rankdir": "LR", "size": "12,8!", "dpi": "150"}
    )
    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
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

    if not save_path:
        save_path = MEDIA_DIR / f"graph_anonymous{now()}"

    dot.render(
        filename=str(save_path),
        format="png",
        cleanup=True,
    )

    Log.info(f"Computation graph saved to {save_path}.png")

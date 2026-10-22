"""Load a Kernel IR json and its launch descriptor into one KernelSpec.

emmy is used here only to deserialize the IR (Graph.from_dict); everything
downstream works on the typed statement tree it returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from emmy.compiler.graph import Graph


@dataclass(frozen=True)
class Param:
    name: str
    dtype: str
    shape: tuple[int, ...]
    scalar: bool = False    # passed by value instead of as a buffer pointer


@dataclass(frozen=True)
class KernelSpec:
    name: str
    body: tuple                 # top-level IR statements
    params: tuple[Param, ...]   # in launch order (meta.json arg_order)
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None
    constants: dict[str, float]  # ConstantOp buffer -> its value (also passed at launch)
    smem_bytes: int = 0          # dynamic shared memory the launch provides (meta.json)

    @property
    def threads_launched(self) -> int | None:
        if self.grid is None or self.block is None:
            return None
        return math.prod(self.grid) * math.prod(self.block)


def _find_kernel_node(graph: Graph, stem: str):
    kernels = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if not kernels:
        raise ValueError("IR contains no KernelOp")
    if len(kernels) == 1:
        return kernels[0]
    # Several kernels in one graph: the file is named after the entry it holds.
    for n in kernels:
        if n.op.name == stem:
            return n
    raise ValueError(f"cannot tell which of {[n.op.name for n in kernels]} to compile")


def _graph_buffers(graph: Graph) -> dict[str, Param]:
    """Shape/dtype of every buffer the graph names (inputs, constants, outputs)."""
    out = {}
    for node in graph.nodes.values():
        for o in node.outputs:
            out[o.name] = Param(o.name, getattr(o.dtype, "name", str(o.dtype)),
                                tuple(int(d.value) for d in o.shape))
    return out


def _constants(graph: Graph) -> dict[str, float]:
    """Scalar ConstantOp values. The IR carries them, so loads of these 1-element
    buffers can be folded to immediates (and pow(x, 3.0) specialized)."""
    out = {}
    for node in graph.nodes.values():
        value = getattr(node.op, "value", None)
        if type(node.op).__name__ == "ConstantOp" and isinstance(value, (int, float)):
            out[node.id] = value
    return out


def load(ir_path: str | Path) -> KernelSpec:
    ir_path = Path(ir_path)
    graph = Graph.from_dict(json.loads(ir_path.read_text(encoding="utf-8")))
    stem = ir_path.name.removesuffix(".ir.json")
    node = _find_kernel_node(graph, stem)
    kernel = node.op

    # meta.json is the contract for parameter order and launch geometry.
    meta_path = ir_path.parent / "meta.json"
    entry = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entry = next((k for k in meta["kernels"] if k["entry"] == kernel.name), None)

    if entry is not None:
        params = tuple(Param(a["name"], a["dtype"], tuple(a["shape"]), a.get("scalar", False))
                       for a in entry["args"])
        by_name = {p.name: p for p in params}
        params = tuple(by_name[n] for n in entry["arg_order"])
        grid, block = tuple(entry["grid"]), tuple(entry["block"])
        smem_bytes = int(entry.get("smem_bytes", 0))
    else:
        # No descriptor: fall back to the node's inputs followed by its outputs.
        names = list(node.inputs) + [o.name for o in node.outputs]
        buffers = _graph_buffers(graph)
        params = tuple(buffers[n] for n in names)
        grid = block = None
        smem_bytes = 0

    return KernelSpec(kernel.name, tuple(kernel.body), params, grid, block, _constants(graph), smem_bytes)

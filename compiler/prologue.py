"""Kernel header, signature and parameter loads.

Everything here is fixed by the launch contract rather than by the IR body:
the PTX header and the parameter list in meta.json order.
"""

from __future__ import annotations

import re

from .context import Buffer, Context, Value
from .loader import KernelSpec

HEADER = ".version 8.0\n.target sm_89\n.address_size 64\n"

# By-value scalar params: meta dtype -> (PTX param type, register kind).
SCALAR_TYPES = {"f32": (".f32", "f32"), "i32": (".s32", "s32"), "i64": (".s64", "s64")}


def param_symbol(name: str) -> str:
    """Buffer names are IR identifiers; prefix them so they never clash with PTX names."""
    return "param_" + re.sub(r"\W", "_", name)


def _scalar_type(p) -> tuple[str, str]:
    try:
        return SCALAR_TYPES[p.dtype]
    except KeyError:
        raise NotImplementedError(f"scalar parameter {p.name!r} of dtype {p.dtype}") from None


def signature(spec: KernelSpec) -> str:
    # Every buffer, constants included, arrives as a .u64 device pointer;
    # only args meta.json marks "scalar" are passed by value.
    params = ",\n".join(
        f"\t.param {_scalar_type(p)[0] if p.scalar else '.u64'} {param_symbol(p.name)}"
        for p in spec.params)
    return f".visible .entry {spec.name}(\n{params}\n)"


def load_params(ctx: Context, spec: KernelSpec) -> None:
    """ld.param each pointer and convert it to the global window once, up front."""
    em = ctx.em
    for p in spec.params:
        if p.scalar:
            ptype, kind = _scalar_type(p)
            r = em.reg(kind)
            em.emit(f"ld.param{ptype} \t{r}, [{param_symbol(p.name)}]", p.name)
            ctx.bind(p.name, Value(r, kind))
            continue
        raw, glob = em.reg("u64"), em.reg("u64")
        em.emit(f"ld.param.u64 \t{raw}, [{param_symbol(p.name)}]")
        em.emit(f"cvta.to.global.u64 \t{glob}, {raw}", p.name)
        ctx.buffers[p.name] = Buffer(p.name, glob, p.dtype, p.shape)

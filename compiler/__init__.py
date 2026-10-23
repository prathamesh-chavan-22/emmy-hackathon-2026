"""Emmy Kernel IR -> PTX compiler."""

from __future__ import annotations

from .context import Context
from .emitter import Emitter
from .loader import KernelSpec, load
from .prologue import HEADER, load_params, signature
from .stmts import lower_body
from .coop import DYN_POOL  # importing coop also registers the cross-thread handlers
from . import mma  # noqa: F401  (registers the tensor-core handlers)


def compile_kernel(spec: KernelSpec) -> str:
    """Lower one kernel to a complete PTX module."""
    ctx = Context(Emitter(), {}, spec.threads_launched, constants=spec.constants,
                  smem_pool=spec.smem_bytes)
    load_params(ctx, spec)
    lower_body(ctx, spec.body)
    ctx.em.emit("ret")

    body = ctx.em.declarations() + [""] + ctx.em.lines
    # The dynamic pool is module-scope; it is declared once, after every Smem
    # has reported the alignment it needs.
    pool = [f".extern .shared .align {ctx.smem_align} .b8 {DYN_POOL}[];"] if ctx.smem_used else []
    ptx = "\n".join([HEADER, *pool, signature(spec), "{", *body, "}", ""])
    ptx.encode("ascii")  # PTX rejects non-ASCII anywhere; fail here, not in ptxas
    return ptx


__all__ = ["compile_kernel", "load"]

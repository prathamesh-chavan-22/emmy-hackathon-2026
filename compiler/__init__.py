"""Emmy Kernel IR -> PTX compiler."""

from __future__ import annotations

from .context import Context
from .emitter import Emitter
from .loader import KernelSpec, load
from .prologue import HEADER, load_params, signature
from .stmts import lower_body
from . import coop  # noqa: F401  registers the cross-thread statement handlers


def compile_kernel(spec: KernelSpec) -> str:
    """Lower one kernel to a complete PTX module."""
    ctx = Context(Emitter(), {}, spec.threads_launched, constants=spec.constants)
    load_params(ctx, spec)
    lower_body(ctx, spec.body)
    ctx.em.emit("ret")

    body = ctx.em.declarations() + [""] + ctx.em.lines
    ptx = "\n".join([HEADER, signature(spec), "{", *body, "}", ""])
    ptx.encode("ascii")  # PTX rejects non-ASCII anywhere; fail here, not in ptxas
    return ptx


__all__ = ["compile_kernel", "load"]

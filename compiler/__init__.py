"""Emmy Kernel IR -> PTX compiler."""

from __future__ import annotations

import re

from .context import Context
from .emitter import Emitter
from .loader import KernelSpec, load
from .prologue import HEADER, load_params, signature
from .stmts import lower_body
from .coop import DYN_POOL  # importing coop also registers the cross-thread handlers
from . import mma  # noqa: F401  (registers the tensor-core handlers)


# Torch ops whose f16 products are summed exactly in f32 (cuBLAS / flash attention).
MATMUL_OPS = re.compile(r"(?:^|_)(?:matmul|mm|bmm|addmm|baddbmm|addbmm|mv|addmv|dot|inner|"
                        r"tensordot|linear|bilinear|sdpa|einsum)(?:_|$)")


def matmul_kernel(name: str) -> bool:
    """Whether the kernel computes a torch matmul. `(a * b).sum(-1)` and a
    vector matmul lower to the same IR (an f16 multiply summed into f32), but
    torch rounds the first one's products to half and not the second's. Only
    the source op tells them apart, and the kernel name is the one place it
    survives (the IR and meta.json carry no op list).

    Limits: a fusion named after its epilogue (`a @ b + c` -> k_add) rounds
    its products like emmy's own C does, and a matmul kernel that also fuses
    an elementwise product-sum keeps that one exact. Both are off by a
    fraction of an output ulp."""
    return bool(MATMUL_OPS.search(name.removeprefix("k_")))


def compile_kernel(spec: KernelSpec) -> str:
    """Lower one kernel to a complete PTX module."""
    ctx = Context(Emitter(), {}, spec.threads_launched, constants=spec.constants,
                  smem_pool=spec.smem_bytes, exact_f16_products=matmul_kernel(spec.name))
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

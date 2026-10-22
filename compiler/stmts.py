"""Statement lowering: one function per IR statement kind, dispatched by class name."""

from __future__ import annotations

import math

from . import expr as X
from .context import Buffer, Context, Value
from .ops import emit_op
from .prologue import global_thread_id

DTYPE_BYTES = {"f32": 4}   # every open problem is f32; others fail loudly, not silently
DTYPE_KIND = {"f32": "f32", "bool": "pred"}   # IR dtype -> register kind


def dtype_name(dt) -> str:
    return getattr(dt, "name", str(dt))


def dtype_bytes(dt: str) -> int:
    try:
        return DTYPE_BYTES[dt]
    except KeyError:
        raise NotImplementedError(f"dtype {dt}") from None


def lower_body(ctx: Context, body) -> None:
    for stmt in body:
        kind = type(stmt).__name__
        try:
            handler = HANDLERS[kind]
        except KeyError:
            raise NotImplementedError(f"statement {kind}") from None
        handler(ctx, stmt)


# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------

def element_address(ctx: Context, buf: Buffer, index) -> tuple[str, int]:
    """Global address of buf[index] as (base register, constant byte offset).

    The IR gives one index expression per buffer dimension; flatten row-major.
    Constant parts fold into the byte offset, which PTX takes for free in
    [reg+imm] addressing - so a vector of unrolled cells shares one base.
    """
    em = ctx.em
    flat = X.imm(0)
    for e, stride in zip(index, buf.strides):
        flat = X._binary(ctx, "+", flat, X._binary(ctx, "*", X.lower(ctx, e), X.imm(stride)))
    size = dtype_bytes(buf.dtype)
    if flat.is_imm:
        return buf.addr, flat.const * size
    wide, addr = em.reg("u64"), em.reg("u64")
    em.emit(f"mul.wide.s32 \t{wide}, {flat.text}, {size}")
    em.emit(f"add.s64 \t{addr}, {buf.addr}, {wide}")
    return addr, 0


def _mem(base: str, offset: int) -> str:
    return f"[{base}+{offset}]" if offset else f"[{base}]"


def _vector_groups(n: int) -> list[tuple[int, int]]:
    """Split n consecutive elements into (start, width) chunks of v4/v2/scalar."""
    out, i = [], 0
    for w in (4, 2, 1):
        while n - i >= w:
            out.append((i, w))
            i += w
    return out


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

def lower_tile(ctx: Context, s) -> None:
    """Map the Tile's N cells onto threads; decode each thread's axis indices.

    The flat cell id is the global thread id. The bounds guard (cell < N) is not
    in the IR - surplus threads in the launch must skip the body. If the launch
    has fewer threads than cells, each thread strides over several cells.
    """
    em = ctx.em
    extents = [int(a.extent.value) for a in s.axes]
    n = math.prod(extents)
    gid = global_thread_id(ctx)
    end = em.label("tile_end")
    strided = ctx.threads_launched is not None and ctx.threads_launched < n

    cell = gid
    if strided:
        cell = Value(em.reg("s32"), "s32")
        em.emit(f"mov.u32 \t{cell.text}, {gid.text}")
        top = em.label("tile_loop")
        em.place(top)
    p = em.reg("pred")
    em.emit(f"setp.ge.s32 \t{p}, {cell.text}, {n}", f"guard: cell < N={n}")
    em.emit(f"@{p} bra \t{end}")

    # Row-major decode: last axis varies fastest.
    rest = cell
    for i in range(len(s.axes) - 1, -1, -1):
        name = s.axes[i].name
        if i == 0:
            ctx.bind(name, rest)
            break
        ctx.bind(name, X._binary(ctx, "%", rest, X.imm(extents[i])))
        rest = X._binary(ctx, "/", rest, X.imm(extents[i]))

    em.comment(f"Tile {[a.name for a in s.axes]} N={n}")
    lower_body(ctx, s.body)

    if strided:
        em.emit(f"add.s32 \t{cell.text}, {cell.text}, {ctx.threads_launched}", "next cell")
        em.emit(f"bra.uni \t{top}")
    em.place(end)


def lower_load(ctx: Context, s) -> None:
    """`a, b, ... = load x[i]`: consecutive elements -> ld.global.v4/v2 where possible."""
    em = ctx.em
    if s.input in ctx.constants and len(s.names) == 1:
        # A ConstantOp arg: its value is in the IR, so no memory access is needed.
        ctx.bind(s.names[0], X.imm(float(ctx.constants[s.input])))
        return
    buf = ctx.buffers[s.input]
    dt = dtype_name(s.dtype)
    base, off = element_address(ctx, buf, s.index)
    regs = [em.reg(dt) for _ in s.names]
    size = dtype_bytes(dt)
    for start, w in _vector_groups(len(regs)):
        dst = regs[start] if w == 1 else "{" + ", ".join(regs[start:start + w]) + "}"
        vec = "" if w == 1 else f".v{w}"
        em.emit(f"ld.global{vec}.{dt} \t{dst}, {_mem(base, off + start * size)}",
                f"{', '.join(s.names[start:start + w])} = {s.input}[...]")
    for name, r in zip(s.names, regs):
        ctx.bind(name, Value(r, dt))


def lower_write(ctx: Context, s) -> None:
    """`buf[i] = (v0, v1, ...)`: vector store, or red.global.add when atomic."""
    em = ctx.em
    buf = ctx.buffers[s.output]
    dt = dtype_name(s.value_dtype)
    base, off = element_address(ctx, buf, s.index)
    vals = [X.to_reg(ctx, ctx.lookup(v)).text for v in s.values]
    size = dtype_bytes(dt)
    if s.atomic:
        # No vector red.add for f32 before sm_90: one reduction per element.
        for i, v in enumerate(vals):
            em.emit(f"red.global.add.{dt} \t{_mem(base, off + i * size)}, {v}")
        return
    for start, w in _vector_groups(len(vals)):
        src = vals[start] if w == 1 else "{" + ", ".join(vals[start:start + w]) + "}"
        vec = "" if w == 1 else f".v{w}"
        em.emit(f"st.global{vec}.{dt} \t{_mem(base, off + start * size)}, {src}",
                f"{s.output}[...] = {', '.join(s.values[start:start + w])}")


def lower_assign(ctx: Context, s) -> None:
    """`name = op(args)`: one elementwise op from the op table.

    Integer operands (index values) are converted to f32; bool values live in
    predicate registers. Constants are passed along so ops like pow can specialize.
    """
    dt = dtype_name(s.dtype)
    try:
        kind = DTYPE_KIND[dt]
    except KeyError:
        raise NotImplementedError(f"Assign of dtype {dt}") from None
    values = [ctx.lookup(a) for a in s.args]
    values = [v if v.kind == "pred" else X.to_f32(ctx, v) for v in values]
    consts = [v.const if v.is_imm else None for v in values]
    args = [X.to_reg(ctx, v).text for v in values]
    dst = ctx.em.reg(kind)
    emit_op(ctx.em, s.op.name, dst, args, consts)
    ctx.bind(s.name, Value(dst, kind))


def lower_let(ctx: Context, s) -> None:
    """`name = expr`: pure expression; constants stay immediates."""
    ctx.bind(s.name, X.lower(ctx, s.value))


HANDLERS = {
    "Tile": lower_tile,
    "Load": lower_load,
    "Write": lower_write,
    "Assign": lower_assign,
    "Let": lower_let,
}

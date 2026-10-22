"""Statement lowering: one function per IR statement kind, dispatched by class name.

This module holds the per-thread statements (Tile, loops, Cond, Load/Write,
Assign/Let/Accum/Init). The cross-thread ones (Smem, Sync, WarpShuffle,
TreeHalve) live in coop.py and register into the same table.
"""

from __future__ import annotations

import math

from . import expr as X
from .context import Buffer, Context, Value
from .ops import emit_op
from .prologue import thread_ids

DTYPE_BYTES = {"f32": 4}   # every open problem is f32; others fail loudly, not silently
DTYPE_KIND = {"f32": "f32", "bool": "pred"}   # IR dtype -> register kind

HANDLERS = {}


def handles(kind: str):
    """Register a lowering function for the IR statement class `kind`."""
    def register(fn):
        HANDLERS[kind] = fn
        return fn
    return register


def lower_body(ctx: Context, body) -> None:
    for stmt in body:
        kind = type(stmt).__name__
        try:
            handler = HANDLERS[kind]
        except KeyError:
            raise NotImplementedError(f"statement {kind}") from None
        handler(ctx, stmt)


def dtype_name(dt) -> str:
    return "f32" if dt is None else getattr(dt, "name", str(dt))


def dtype_bytes(dt: str) -> int:
    try:
        return DTYPE_BYTES[dt]
    except KeyError:
        raise NotImplementedError(f"dtype {dt}") from None


# ---------------------------------------------------------------------------
# Addressing and memory access
# ---------------------------------------------------------------------------

def flat_index(ctx: Context, buf: Buffer, index) -> Value:
    """Row-major element offset of buf[index]; the IR gives one expression per dim."""
    flat = X.imm(0)
    for e, stride in zip(index, buf.strides):
        flat = X._binary(ctx, "+", flat, X._binary(ctx, "*", X.lower(ctx, e), X.imm(stride)))
    return flat


def address(ctx: Context, buf: Buffer, flat: Value) -> tuple[str, int]:
    """Address of element `flat` as (base register, constant byte offset).

    A constant offset goes into PTX's free [reg+imm] addressing, so unrolled
    cells share one base. Global addresses are 64-bit; shared ones are 32-bit
    offsets into the shared window.
    """
    em = ctx.em
    size = dtype_bytes(buf.dtype)
    if flat.is_imm:
        return buf.addr, flat.const * size
    if buf.space == "shared":
        off, addr = em.reg("s32"), em.reg("s32")
        em.emit(f"mul.lo.s32 \t{off}, {flat.text}, {size}")
        em.emit(f"add.s32 \t{addr}, {buf.addr}, {off}")
        return addr, 0
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


def load_values(ctx: Context, buf: Buffer, flat: Value, count: int, names=()) -> list[Value]:
    """Load `count` consecutive elements starting at `flat`, vectorized (v4/v2) where possible."""
    em = ctx.em
    dt = buf.dtype
    base, off = address(ctx, buf, flat)
    regs = [em.reg(dt) for _ in range(count)]
    size = dtype_bytes(dt)
    for start, w in _vector_groups(count):
        dst = regs[start] if w == 1 else "{" + ", ".join(regs[start:start + w]) + "}"
        vec = "" if w == 1 else f".v{w}"
        note = f"{', '.join(names[start:start + w])} = {buf.name}[...]" if names else None
        em.emit(f"ld.{buf.space}{vec}.{dt} \t{dst}, {_mem(base, off + start * size)}", note)
    return [Value(r, dt) for r in regs]


def store_values(ctx: Context, buf: Buffer, flat: Value, values: list[Value], atomic=False, names=()) -> None:
    """Store consecutive elements: vector st, or one red.add per element when atomic."""
    em = ctx.em
    dt = buf.dtype
    base, off = address(ctx, buf, flat)
    vals = [X.to_reg(ctx, v).text for v in values]
    size = dtype_bytes(dt)
    if atomic:
        # No vector red.add for f32 before sm_90: one reduction per element.
        for i, v in enumerate(vals):
            em.emit(f"red.{buf.space}.add.{dt} \t{_mem(base, off + i * size)}, {v}")
        return
    for start, w in _vector_groups(len(vals)):
        src = vals[start] if w == 1 else "{" + ", ".join(vals[start:start + w]) + "}"
        vec = "" if w == 1 else f".v{w}"
        note = f"{buf.name}[...] = {', '.join(names[start:start + w])}" if names else None
        em.emit(f"st.{buf.space}{vec}.{dt} \t{_mem(base, off + start * size)}, {src}", note)


# ---------------------------------------------------------------------------
# Structure: Tile, loops, Cond
# ---------------------------------------------------------------------------

@handles("Tile")
def lower_tile(ctx: Context, s) -> None:
    """Map the Tile's N cells onto threads; decode each thread's axis indices.

    The flat cell id is the global thread id. The bounds guard (cell < N) is not
    in the IR - surplus threads in the launch must skip the body. If the launch
    has fewer threads than cells, each thread strides over several cells.
    `lane` and `warp` are the names cooperative statements use for the thread's
    position in its warp and the warp's index in the block.
    """
    em = ctx.em
    extents = [int(a.extent.value) for a in s.axes]
    n = math.prod(extents)
    tid, gid = thread_ids(ctx)
    lane, warp = em.reg("s32"), em.reg("s32")
    em.emit(f"and.b32 \t{lane}, {tid.text}, 31", "lane")
    em.emit(f"shr.u32 \t{warp}, {tid.text}, 5", "warp")
    ctx.bind("lane", Value(lane, "s32"))
    ctx.bind("warp", Value(warp, "s32"))

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


def _counted_loop(ctx: Context, s, start: Value, step: Value, end: Value) -> None:
    """for (axis = start; axis < end; axis += step) { body }, shared by Loop and StridedLoop.

    Accumulators folded directly in the body start at their op's identity just
    before the loop (unless `seed` is off: then an enclosing Init owns them).
    """
    em = ctx.em
    if s.seed:
        for acc in s.body:
            if type(acc).__name__ == "Accum" and acc.name not in ctx.mutable:
                ctx.declare_var(acc.name, X.imm(float(acc.op.identity)))
    i = ctx.declare_var(s.axis.name, start)
    top, done = em.label(f"loop_{s.axis.name}"), em.label(f"loop_{s.axis.name}_end")
    p = em.reg("pred")
    em.place(top)
    em.emit(f"setp.ge.s32 \t{p}, {i.text}, {end.text}", f"{s.axis.name} < {end.text}")
    em.emit(f"@{p} bra \t{done}")
    lower_body(ctx, s.body)
    em.emit(f"add.s32 \t{i.text}, {i.text}, {step.text}")
    em.emit(f"bra.uni \t{top}")
    em.place(done)
    # Seeded accumulators finish here; a later loop may seed the same name again.
    for acc in s.body:
        if type(acc).__name__ == "Accum" and s.seed:
            ctx.mutable.pop(acc.name, None)


@handles("StridedLoop")
def lower_strided_loop(ctx: Context, s) -> None:
    """Cooperative loop: thread t visits start, start+step, ... below the axis extent."""
    end = X.lower(ctx, s.end) if s.end is not None else X.imm(int(s.axis.extent.value))
    if s.end is not None:
        ctx.bind(f"{s.axis.name}_end", end)   # emmy lets the body reference the hoisted bound
    _counted_loop(ctx, s, X.lower(ctx, s.start), X.lower(ctx, s.step), end)


@handles("Loop")
def lower_loop(ctx: Context, s) -> None:
    """Serial loop over the whole axis: 0, 1, ..., extent-1."""
    _counted_loop(ctx, s, X.imm(0), X.imm(1), X.imm(int(s.axis.extent.value)))


@handles("Cond")
def lower_cond(ctx: Context, s) -> None:
    """if (cond) body else else_body: branch around each side on the predicate."""
    em = ctx.em
    p = X.lower(ctx, s.cond)
    if p.is_imm:                      # statically decided: emit only the live side
        lower_body(ctx, s.body if p.const else s.else_body)
        return
    if p.kind != "pred":
        p = X._binary(ctx, "!=", p, X.imm(0))
    end = em.label("endif")
    skip = em.label("else") if s.else_body else end
    em.emit(f"@!{p.text} bra \t{skip}")
    lower_body(ctx, s.body)
    if s.else_body:
        em.emit(f"bra.uni \t{end}")
        em.place(skip)
        lower_body(ctx, s.else_body)
    em.place(end)


# ---------------------------------------------------------------------------
# Memory and values
# ---------------------------------------------------------------------------

@handles("Load")
def lower_load(ctx: Context, s) -> None:
    """`a, b, ... = load x[i]`: consecutive elements -> ld.v4/v2 where possible."""
    if s.input in ctx.constants and len(s.names) == 1:
        # A ConstantOp arg: its value is in the IR, so no memory access is needed.
        ctx.assign(s.names[0], X.imm(float(ctx.constants[s.input])))
        return
    buf = ctx.buffers[s.input]
    values = load_values(ctx, buf, flat_index(ctx, buf, s.index), len(s.names), s.names)
    for name, v in zip(s.names, values):
        ctx.assign(name, v)


@handles("Write")
def lower_write(ctx: Context, s) -> None:
    """`buf[i] = (v0, v1, ...)`: vector store, or red.add when atomic."""
    buf = ctx.buffers[s.output]
    values = [ctx.lookup(v) for v in s.values]
    store_values(ctx, buf, flat_index(ctx, buf, s.index), values, s.atomic, s.values)


def _operands(ctx: Context, names) -> tuple[list[str], list]:
    """Argument registers for an op (ints promoted to f32, bools kept as .pred) and their constants."""
    values = [ctx.lookup(a) for a in names]
    values = [v if v.kind == "pred" else X.to_f32(ctx, v) for v in values]
    return [X.to_reg(ctx, v).text for v in values], [v.const if v.is_imm else None for v in values]


@handles("Assign")
def lower_assign(ctx: Context, s) -> None:
    """`name = op(args)`: one elementwise op from the op table.

    Integer operands (index values) are converted to f32; bool values live in
    predicate registers. Constants are passed along so ops like pow can specialize.
    The combine programs inside WarpShuffle/TreeHalve leave dtype unset: f32.
    """
    dt = dtype_name(s.dtype)
    try:
        kind = DTYPE_KIND[dt]
    except KeyError:
        raise NotImplementedError(f"Assign of dtype {dt}") from None
    args, consts = _operands(ctx, s.args)
    dst = ctx.em.reg(kind)
    emit_op(ctx.em, s.op.name, dst, args, consts)
    ctx.assign(s.name, Value(dst, kind))


@handles("Accum")
def lower_accum(ctx: Context, s) -> None:
    """`acc <- op(acc, value)` folded into the accumulator's own register.

    Ops write their destination last, so the fold can target the accumulator
    register directly. `base` (when set) replaces acc as the left operand.
    """
    acc = ctx.mutable.get(s.name)
    if acc is None:
        raise ValueError(f"Accum {s.name!r} has no seeded accumulator in scope")
    args, consts = _operands(ctx, [s.base or s.name, s.value])
    emit_op(ctx.em, s.op.name, acc.text, args, consts)


@handles("Init")
def lower_init(ctx: Context, s) -> None:
    """Carried state seeded with a literal; later folds update it in place."""
    ctx.declare_var(s.name, X.imm(float(s.identity)))


@handles("Let")
def lower_let(ctx: Context, s) -> None:
    """`name = expr`: pure expression; constants stay immediates."""
    ctx.assign(s.name, X.lower(ctx, s.value))

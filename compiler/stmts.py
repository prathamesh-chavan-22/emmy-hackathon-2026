"""Statement lowering: one function per IR statement kind, dispatched by class name.

This module holds the per-thread statements (Tile, loops, Cond, Load/Write,
Assign/Let/Accum/Init). The cross-thread ones (Smem, cp.async, Sync,
WarpShuffle, TreeHalve) live in coop.py and register into the same table.
Address math and vector loads/stores live in memory.py.
"""

from __future__ import annotations

import math

from . import expr as X
from .context import Context, Value
from .memory import flat_index, load_values, store_values
from .ops import emit_op, signature

DTYPE_KIND = {"f32": "f32", "bool": "pred"}   # IR dtype -> register kind
PARTIAL = 8    # copies per trip for innermost loops without emmy's unroll flag
FULL_UNROLL_MAX = 64   # emmy marks extents up to 64; anything longer gets PARTIAL instead

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


def lower_block(ctx: Context, body) -> None:
    """A nested statement list (an if/else side): names it binds stay inside it.

    Writes to carried state still land, since those go into the state's own register.
    """
    with ctx.scope():
        lower_body(ctx, body)


def dtype_name(dt) -> str:
    return "f32" if dt is None else getattr(dt, "name", str(dt))


# ---------------------------------------------------------------------------
# Structure: Tile, loops, Cond
# ---------------------------------------------------------------------------

def thread_ids(ctx: Context) -> tuple[Value, Value]:
    """(tid.x, ctaid.x * ntid.x + tid.x): block-local id for cooperative code,
    global id = the flat cell this thread starts at (1-D grids)."""
    em = ctx.em
    tid, ctaid, ntid, gid = (em.reg("s32") for _ in range(4))
    em.emit(f"mov.u32 \t{tid}, %tid.x")
    em.emit(f"mov.u32 \t{ctaid}, %ctaid.x")
    em.emit(f"mov.u32 \t{ntid}, %ntid.x")
    em.emit(f"mad.lo.s32 \t{gid}, {ctaid}, {ntid}, {tid}", "global thread id")
    ctx.bind("%tid", Value(tid, "s32"))
    return Value(tid, "s32"), Value(gid, "s32")


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
        ctx.bind(name, X.binary(ctx, "%", rest, X.imm(extents[i])))
        rest = X.binary(ctx, "/", rest, X.imm(extents[i]))

    em.comment(f"Tile {[a.name for a in s.axes]} N={n}")
    lower_body(ctx, s.body)

    if strided:
        em.emit(f"add.s32 \t{cell.text}, {cell.text}, {ctx.threads_launched}", "next cell")
        em.emit(f"bra.uni \t{top}")
    em.place(end)


def _seed_accumulators(ctx: Context, s) -> None:
    """Start each accumulator the loop body folds into at its op's identity (when `seed`)."""
    if not s.seed:
        return
    seeded = set()
    for acc in s.body:
        if type(acc).__name__ == "Accum" and acc.name not in seeded:
            ctx.declare_var(acc.name, X.imm(float(acc.op.identity)))
            seeded.add(acc.name)


def _counted_loop(ctx: Context, s, start: Value, step: Value, end: Value, copies: int = 1) -> None:
    """for (axis = start; axis < end; axis += step) { body }, shared by Loop and StridedLoop.

    Accumulators folded directly in the body start at their op's identity just
    before the loop, in a fresh register that shadows any outer same-named state
    (emmy's `T acc = identity;`). With `seed` off an enclosing Init owns them.
    The body is a scope: only this loop's accumulators outlive it.

    `copies` > 1 (constant step only) first runs the body `copies` times per trip
    while all of them are in range, copy k seeing the axis as `axis + k*step`,
    then finishes the remainder one at a time. Each thread performs exactly the
    same operations in the same order, so results are bit-identical.
    """
    em = ctx.em
    _seed_accumulators(ctx, s)
    # Seeded accumulators were declared outside the body's scope, so they live on
    # in the enclosing block (as in emmy's C), still writable.
    with ctx.scope():
        i = ctx.declare_var(s.axis.name, start)
        name = s.axis.name
        if copies > 1:
            # No remainder when the trip count is static and divisible: then the
            # last copy is in range whenever the first is.
            exact = end.is_imm and start.is_imm and (end.const - start.const) % (copies * step.const) == 0
            last = 0 if exact else (copies - 1) * step.const
            top, done = em.label(f"loop_{name}_x{copies}"), em.label(f"loop_{name}_x{copies}_end")
            em.place(top)
            p = X.binary(ctx, ">=", X.binary(ctx, "+", i, X.imm(last)), end)
            em.emit(f"@{p.text} bra \t{done}", f"all {copies} of {name}..{name}+{last} < {end.text}?")
            for k in range(copies):
                with ctx.scope():
                    off = k * step.const
                    ctx.bind(name, X.binary(ctx, "+", i, X.imm(off)))
                    ctx.offsets[name] = (i, off)
                    lower_body(ctx, s.body)
            em.emit(f"add.s32 \t{i.text}, {i.text}, {copies * step.const}")
            em.emit(f"bra.uni \t{top}")
            em.place(done)
            if exact:
                return
        top, done = em.label(f"loop_{name}"), em.label(f"loop_{name}_end")
        p = em.reg("pred")
        em.place(top)
        em.emit(f"setp.ge.s32 \t{p}, {i.text}, {end.text}", f"{name} < {end.text}")
        em.emit(f"@{p} bra \t{done}")
        lower_body(ctx, s.body)
        em.emit(f"add.s32 \t{i.text}, {i.text}, {step.text}")
        em.emit(f"bra.uni \t{top}")
        em.place(done)


def _strided_copies(s, step: Value, end: Value) -> int:
    """Body copies per trip for a StridedLoop (1 = rolled); see lower_loop.

    Needs a constant step, and enough trips (start is below one step, so about
    end/step of them) for a full group of copies to run at least twice.
    """
    if not (step.is_imm and step.const > 0) or _has_loop(s.body):
        return 1
    if end.is_imm and end.const < 2 * PARTIAL * step.const:
        return 1
    return PARTIAL


def _has_loop(body) -> bool:
    for b in body:
        kind = type(b).__name__
        if kind in ("Loop", "StridedLoop"):
            return True
        if kind == "Cond" and (_has_loop(b.body) or _has_loop(b.else_body or ())):
            return True
    return False


@handles("StridedLoop")
def lower_strided_loop(ctx: Context, s) -> None:
    """Cooperative loop: thread t visits start, start+step, ... below the axis extent."""
    end = X.lower(ctx, s.end) if s.end is not None else X.imm(int(s.axis.extent.value))
    if s.end is not None:
        ctx.bind(f"{s.axis.name}_end", end)   # emmy lets the body reference the hoisted bound
    step = X.lower(ctx, s.step)
    _counted_loop(ctx, s, X.lower(ctx, s.start), step, end, _strided_copies(s, step, end))


@handles("Loop")
def lower_loop(ctx: Context, s) -> None:
    """Serial loop over the whole axis: 0, 1, ..., extent-1.

    Innermost loops are unrolled, as nvcc does for emmy's C:
    - `unroll` (emmy's `#pragma unroll`, set on hot loops such as a matmul's
      k-step): fully, the axis a constant, so index math folds into immediate
      address offsets.
    - otherwise PARTIAL copies per trip (plus a remainder loop), so several loads
      are in flight per trip instead of one latency-bound chain. Strided loops
      get the same treatment.
    A loop holding other loops stays rolled: copying whole loops bought nothing
    and ran attention out of registers (255, spilling: 5x slower).
    """
    extent = int(s.axis.extent.value)
    if _has_loop(s.body):
        _counted_loop(ctx, s, X.imm(0), X.imm(1), X.imm(extent))
    elif s.unroll and extent <= FULL_UNROLL_MAX:
        _seed_accumulators(ctx, s)
        for i in range(extent):
            with ctx.scope():
                ctx.bind(s.axis.name, X.imm(i))
                lower_body(ctx, s.body)
    else:
        copies = PARTIAL if extent >= 2 * PARTIAL else 1
        _counted_loop(ctx, s, X.imm(0), X.imm(1), X.imm(extent), copies)


@handles("Cond")
def lower_cond(ctx: Context, s) -> None:
    """if (cond) body else else_body: branch around each side on the predicate."""
    em = ctx.em
    p = X.lower(ctx, s.cond)
    if p.is_imm:                      # statically decided: emit only the live side
        lower_block(ctx, s.body if p.const else s.else_body)
        return
    if p.kind != "pred":
        p = X.binary(ctx, "!=", p, X.imm(0))
    end = em.label("endif")
    skip = em.label("else") if s.else_body else end
    em.emit(f"@!{p.text} bra \t{skip}")
    lower_block(ctx, s.body)
    if s.else_body:
        em.emit(f"bra.uni \t{end}")
        em.place(skip)
        lower_block(ctx, s.else_body)
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
    # A bool value stored to a float buffer is 1.0/0.0, as in emmy's C.
    values = [X.convert(ctx, ctx.lookup(v), DTYPE_KIND[buf.dtype]) for v in s.values]
    store_values(ctx, buf, flat_index(ctx, buf, s.index), values, s.atomic, s.values)


def _operands(ctx: Context, names, kinds) -> tuple[list[str], list]:
    """Argument registers for an op, each converted to the kind the op computes in, and their constants."""
    values = [X.convert(ctx, ctx.lookup(a), k) for a, k in zip(names, kinds)]
    return [X.to_reg(ctx, v).text for v in values], [v.const if v.is_imm else None for v in values]


@handles("Assign")
def lower_assign(ctx: Context, s) -> None:
    """`name = op(args)`: one elementwise op from the op table.

    Operands are converted to the kinds the op computes in (index ints and
    bools become f32, numbers used as masks become predicates), and the result
    to the IR's dtype: bool lives in a predicate register, f32 in a float one.
    Constants are passed along so ops like pow can specialize.
    The combine programs inside WarpShuffle/TreeHalve leave dtype unset: f32.
    """
    dt = dtype_name(s.dtype)
    try:
        kind = DTYPE_KIND[dt]
    except KeyError:
        raise NotImplementedError(f"Assign of dtype {dt}") from None
    arg_kinds, out_kind = signature(s.op.name, len(s.args))
    args, consts = _operands(ctx, s.args, arg_kinds)
    dst = ctx.em.reg(out_kind)
    emit_op(ctx.em, s.op.name, dst, args, consts)
    if s.op.name == "multiply":
        ctx.products[dst] = (args[0], args[1])
    ctx.assign(s.name, X.convert(ctx, Value(dst, out_kind), kind))


@handles("Accum")
def lower_accum(ctx: Context, s) -> None:
    """`acc <- op(acc, value)` folded into the accumulator's own register.

    Ops write their destination last, so the fold can target the accumulator
    register directly. `base` (when set) replaces acc as the left operand.
    `acc + a*b` (a matmul or dot-product step) becomes one fma.rn, as nvcc
    contracts it in emmy's C - unless a or b is updated in place, since the
    fma re-reads them here, after the multiply.
    """
    acc = ctx.mutable.get(s.name)
    if acc is None:
        raise ValueError(f"Accum {s.name!r} has no seeded accumulator in scope")
    args, consts = _operands(ctx, [s.base or s.name, s.value], ["f32", "f32"])
    product = ctx.products.get(args[1]) if s.op.name == "add" else None
    in_place = {v.text for v in ctx.mutable.values()}
    if product and not in_place.intersection(product):
        ctx.em.emit(f"fma.rn.f32 \t{acc.text}, {product[0]}, {product[1]}, {args[0]}")
        return
    emit_op(ctx.em, s.op.name, acc.text, args, consts)


@handles("Init")
def lower_init(ctx: Context, s) -> None:
    """Carried state seeded with a literal; later folds update it in place."""
    ctx.declare_var(s.name, X.imm(float(s.identity)))


@handles("Select")
def lower_select_stmt(ctx: Context, s) -> None:
    """`name = v0 when p0, v1 when p1, ..., vN`: a chained selp, first true predicate wins.

    The last branch is the catch-all (its predicate is not tested), as in emmy's
    ternary chain. Used for masked tails: out-of-range lanes pick the identity.
    """
    value = ctx.lookup(s.branches[-1].value)
    for b in reversed(s.branches[:-1]):
        value = X.select(ctx, X.lower(ctx, b.select), ctx.lookup(b.value), value)
    ctx.assign(s.name, value)


@handles("Let")
def lower_let(ctx: Context, s) -> None:
    """`name = expr`: pure expression; constants stay immediates."""
    ctx.assign(s.name, X.lower(ctx, s.value))

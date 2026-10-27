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
from .dtypes import DType, INT_KINDS, cast_const, common_kind, get, round_to
from .memory import flat_index, load_values, store_values, swizzled
from .ops import COMPARISONS, SHIFTS, emit_native, emit_op, native, signature

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


def value_dtype(dt) -> DType | None:
    """The IR dtype of a statement's result; None for bool (a predicate)."""
    name = dtype_name(dt)
    return None if name == "bool" else get(name)


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
            ctx.declare_var(acc.name, X.const(float(acc.op.identity), value_dtype(acc.dtype)))
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


def _has_loop(body, rolled_only: bool = False) -> bool:
    """Whether body holds a loop. rolled_only skips fragment loops, which are
    always unrolled: a loop around them can still unroll fully (emmy's
    `#pragma unroll` over a k-step of mma blocks), but gets no partial copies."""
    for b in body:
        kind = type(b).__name__
        if kind in ("Loop", "StridedLoop"):
            if not (rolled_only and b.axis.name in _fragment_index_vars(b.body)):
                return True
            if _has_loop(b.body, rolled_only):   # a rolled loop inside the unrolled one
                return True
        if kind == "Cond" and (_has_loop(b.body, rolled_only)
                               or _has_loop(b.else_body or (), rolled_only)):
            return True
    return False


def _fragment_index_vars(body) -> set[str]:
    """Names used to pick a member of a tensor-core fragment family, as in `_c0_[_r0]`.

    Fragments are registers, which cannot be indexed at run time, so a loop
    over such a name must be unrolled (mma.py).
    """
    out = set()

    def scan(v):
        if isinstance(v, str):
            stem, _, rest = v.partition("[")
            if rest.endswith("]") and stem.isidentifier():
                out.add(rest[:-1])
        elif isinstance(v, (tuple, list)):
            for x in v:
                scan(x)

    for b in body:
        for name in getattr(b, "__dataclass_fields__", ()):
            v = getattr(b, name)
            if name in ("body", "else_body"):
                out |= _fragment_index_vars(v or ())
            else:
                scan(v)
    return out


def _unroll(ctx: Context, s, values) -> None:
    """The body once per axis value, the axis a constant in each copy."""
    _seed_accumulators(ctx, s)
    for v in values:
        with ctx.scope():
            ctx.bind(s.axis.name, X.imm(v))
            lower_body(ctx, s.body)


@handles("StridedLoop")
def lower_strided_loop(ctx: Context, s) -> None:
    """Cooperative loop: thread t visits start, start+step, ... below the axis extent.

    With constant bounds and emmy's unroll flag (or an axis that picks a
    fragment member) it is unrolled like a Loop.
    """
    end = X.lower(ctx, s.end) if s.end is not None else X.imm(int(s.axis.extent.value))
    if s.end is not None:
        ctx.bind(f"{s.axis.name}_end", end)   # emmy lets the body reference the hoisted bound
    start, step = X.lower(ctx, s.start), X.lower(ctx, s.step)
    picks_fragment = s.axis.name in _fragment_index_vars(s.body)
    if start.is_imm and step.is_imm and end.is_imm and step.const > 0:
        values = range(start.const, end.const, step.const)
        if picks_fragment or (s.unroll and len(values) <= FULL_UNROLL_MAX
                               and not _has_loop(s.body, rolled_only=True)):
            _unroll(ctx, s, values)
            return
    if picks_fragment:
        raise NotImplementedError(f"loop {s.axis.name} picks fragments but has runtime bounds")
    _counted_loop(ctx, s, start, step, end, _strided_copies(s, step, end))


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
    and ran attention out of registers (255, spilling: 5x slower). The one
    exception is emmy's `unroll` over fragment loops only (a k-step of mma
    blocks), which nvcc unrolls fully too.
    """
    extent = int(s.axis.extent.value)
    if s.axis.name in _fragment_index_vars(s.body):
        _unroll(ctx, s, range(extent))
    elif s.unroll and extent <= FULL_UNROLL_MAX and not _has_loop(s.body, rolled_only=True):
        _unroll(ctx, s, range(extent))
    elif _has_loop(s.body):
        _counted_loop(ctx, s, X.imm(0), X.imm(1), X.imm(extent))
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
        # It is a Python scalar, which torch keeps at f32 precision even next to an
        # f16/bf16 tensor (x * 0.1 is f16(float(x) * 0.1f)): converted to the compute
        # kind, not rounded to the buffer's narrow type. The result rounds on assignment.
        kind = get(ctx.buffers[s.input].dtype).kind
        ctx.assign(s.names[0], X.imm(cast_const(ctx.constants[s.input], kind), kind))
        return
    buf = ctx.buffers[s.input]
    values = load_values(ctx, buf, flat_index(ctx, buf, s.index), len(s.names), s.names)
    for name, v in zip(s.names, values):
        ctx.assign(name, v)


@handles("Write")
def lower_write(ctx: Context, s) -> None:
    """`buf[i] = (v0, v1, ...)`: vector store, or red.add when atomic.

    A write into a swizzled smem slab (an mma operand computed in place, not
    copied) moves the vector's first index like a cp.async fill would; the
    rest of an 8-half chunk follows it, as the XOR keeps chunks whole.
    """
    buf = ctx.buffers[s.output]
    values = [ctx.lookup(v) for v in s.values]
    flat = swizzled(ctx, flat_index(ctx, buf, s.index), getattr(s, "swizzle", "NONE"))
    store_values(ctx, buf, flat, values, s.atomic, s.values)


def _operands(ctx: Context, names, kinds) -> tuple[list[str], list]:
    """Argument registers for an op, each converted to the kind the op computes in, and their constants."""
    values = [X.convert(ctx, ctx.lookup(a), k) for a, k in zip(names, kinds)]
    return [X.to_reg(ctx, v).text for v in values], [v.const if v.is_imm else None for v in values]


def compute(ctx: Context, op: str, names, dt) -> Value:
    """`op(names...)` with the result typed as IR dtype `dt` (see dtypes.py).

    An int or f64 result uses the native op when there is one (integer
    operands only, for an int: emmy's C does int math only then), and
    integers and doubles compare in their own kind. Everything else computes
    in f32: operands are converted to the kinds the op works in (ints and
    bools become f32, numbers used as masks become predicates), and the
    result to dt's kind - truncated for an int, rounded to half precision for
    f16. Constants are passed along so ops like pow can specialize.
    """
    em = ctx.em
    dtype = value_dtype(dt)
    kind = dtype.kind if dtype else "pred"
    values = [ctx.lookup(n) for n in names]
    operand_kinds = {v.kind for v in values}
    if op in COMPARISONS:
        work = common_kind(operand_kinds) if "pred" not in operand_kinds else "f32"
    elif native(op, kind) and (kind == "f64" or operand_kinds <= set(INT_KINDS)):
        work = kind
    else:
        work = "f32"
    if work != "f32":
        kinds = [work] * len(values)
        if op in SHIFTS:
            kinds[1] = "s32"
        args = [X.to_reg(ctx, X.convert(ctx, v, k)).text for v, k in zip(values, kinds)]
        out = "pred" if op in COMPARISONS else work
        dst = Value(em.reg(out), out)
        emit_native(em, op, work, dst.text, args)
        return X.convert(ctx, dst, kind)

    arg_kinds, out_kind = signature(op, len(names))
    args, consts = _operands(ctx, names, arg_kinds)
    dst = em.reg(out_kind)
    emit_op(em, op, dst, args, consts)
    result = X.convert(ctx, Value(dst, out_kind), kind)
    if dtype is not None and dtype.rounds:
        rounded = Value(round_to(em, result.text, dtype), kind)
        if op == "multiply" and ctx.exact_f16_products:
            # In a matmul torch keeps the f16 product exact, so an Accum that
            # sums it reads the operands into its fma, skipping this rounding.
            ctx.products[rounded.text] = (args[0], args[1])
        return rounded
    if op == "multiply" and result.text == dst:
        ctx.products[dst] = (args[0], args[1])
    return result


@handles("Assign")
def lower_assign(ctx: Context, s) -> None:
    """`name = op(args)`: one elementwise op, typed by the IR's dtype (see compute).
    The combine programs inside WarpShuffle/TreeHalve leave dtype unset: f32."""
    ctx.assign(s.name, compute(ctx, s.op.name, s.args, s.dtype))


@handles("Accum")
def lower_accum(ctx: Context, s) -> None:
    """`acc <- op(acc, value)` folded into the accumulator's own register.

    Ops write their destination last, so an f32 fold can target the
    accumulator register directly. `base` (when set) replaces acc as the left
    operand. `acc + a*b` (a matmul or dot-product step) becomes one fma.rn, as
    nvcc contracts it in emmy's C - unless a or b is updated in place, since
    the fma re-reads them here, after the multiply. An f16 product is
    contracted unrounded in a matmul kernel: emmy's C rounds it to half first,
    but torch's f16 matmul (the expected output) sums exact products in f32. Other accumulator dtypes
    (an f16 max, an int max) compute like an Assign and convert on the way in.
    """
    acc = ctx.mutable.get(s.name)
    if acc is None:
        raise ValueError(f"Accum {s.name!r} has no seeded accumulator in scope")
    left = s.base or s.name
    if dtype_name(s.dtype) != "f32":
        ctx.assign(s.name, compute(ctx, s.op.name, [left, s.value], s.dtype))
        return
    args, consts = _operands(ctx, [left, s.value], ["f32", "f32"])
    product = ctx.products.get(args[1]) if s.op.name == "add" else None
    in_place = {v.text for v in ctx.mutable.values()}
    if product and not in_place.intersection(product):
        ctx.em.emit(f"fma.rn.f32 \t{acc.text}, {product[0]}, {product[1]}, {args[0]}")
        return
    emit_op(ctx.em, s.op.name, acc.text, args, consts)


@handles("Init")
def lower_init(ctx: Context, s) -> None:
    """Carried state seeded with a literal; later folds update it in place."""
    ctx.declare_var(s.name, X.const(float(s.identity), value_dtype(s.dtype)))


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

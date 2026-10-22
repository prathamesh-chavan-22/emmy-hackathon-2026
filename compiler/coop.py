"""Cross-thread statements: shared memory, barriers, warp shuffles, smem trees.

These are what turn per-thread partials into a block-wide result. They
register into the same dispatch table as the per-thread statements.
"""

from __future__ import annotations

import math

from . import expr as X
from .context import Buffer, Context, Value
from .stmts import handles, load_values, lower_body, store_values

SMEM_DTYPES = {"float": "f32"}   # Smem carries the C type name


def _sync(ctx: Context, barrier_id: int, count) -> None:
    """bar.sync 0 = __syncthreads; a named barrier waits for `count` threads only."""
    if barrier_id == 0:
        ctx.em.emit("bar.sync \t0")
    else:
        ctx.em.emit(f"bar.sync \t{barrier_id}, {count}")


@handles("Smem")
def lower_smem(ctx: Context, s) -> None:
    """A per-block .shared array; its address register makes it a normal Buffer.

    Shared addresses are 32-bit offsets in the shared window, so Load/Write on
    it use 32-bit math and ld/st.shared.
    """
    try:
        dt = SMEM_DTYPES[s.dtype]
    except KeyError:
        raise NotImplementedError(f"Smem dtype {s.dtype}") from None
    extents = tuple(int(e) for e in s.extents)
    symbol = f"sh_{s.name}"
    nbytes = max(1, math.prod(extents)) * 4
    ctx.em.declare_shared(symbol, nbytes, max(int(s.align), 16))   # 16: room for v4 access
    base = ctx.em.reg("s32")
    ctx.em.emit(f"mov.u32 \t{base}, {symbol}", s.name)
    ctx.buffers[s.name] = Buffer(s.name, base, dt, extents, space="shared")


@handles("Sync")
def lower_sync(ctx: Context, s) -> None:
    """Barrier: block-wide, named (subset of threads), or warp-only."""
    if s.warp:
        ctx.em.emit("bar.warp.sync \t-1")
    else:
        _sync(ctx, s.barrier_id, s.count)


@handles("WarpShuffle")
def lower_warp_shuffle(ctx: Context, s) -> None:
    """XOR butterfly over `length` lanes: after log2(length) steps every lane holds the total.

    Each step pulls the partner lane's state into state_b and folds it in with
    the carrier's combine program. The member mask is activemask (not 0xffffffff)
    because a small block can be a partial warp.
    """
    em = ctx.em
    mask = em.reg("b32")
    em.emit(f"activemask.b32 \t{mask}")
    step = int(s.length) // 2
    while step > 0:
        for st, sb in zip(s.state, s.state_b):
            src = X.to_reg(ctx, ctx.lookup(st))
            dst = em.reg("f32")
            em.emit(f"shfl.sync.bfly.b32 \t{dst}, {src.text}, {step}, 31, {mask}", f"{sb} = lane ^ {step}")
            ctx.bind(sb, Value(dst, "f32"))
        lower_body(ctx, s.combine_states)
        step //= 2


@handles("TreeHalve")
def lower_tree_halve(ctx: Context, s) -> None:
    """Halve a shared-memory slab per level until slot 0 holds the result, then broadcast.

    Level s: threads t < s fold buf[t+s] into buf[t]; a barrier separates levels.
    The levels are unrolled (length is a compile-time power of two). The
    combine program reassigns the state names, so they are bound to the loaded
    slots for the duration of each level and restored after.
    """
    em = ctx.em
    t = ctx.lookup(s.tid_var)
    bufs = [ctx.buffers[b] for b in s.bufs]
    if s.inner is not None:
        iv, scale = ctx.lookup(s.inner[0]), int(s.inner[1])
        slot_of = lambda k: X._binary(ctx, "+", X._binary(ctx, "*", k, X.imm(scale)), iv)
        root = iv
    else:
        slot_of = lambda k: k
        root = X.imm(0)

    saved = {st: ctx.mutable.pop(st, None) for st in s.state}
    half = int(s.length) // 2
    while half > 0:
        skip = em.label("tree_skip")
        p = em.reg("pred")
        em.emit(f"setp.ge.s32 \t{p}, {X.to_reg(ctx, t).text}, {half}", f"tree level {half}")
        em.emit(f"@{p} bra \t{skip}")
        mine = slot_of(t)
        other = slot_of(X._binary(ctx, "+", t, X.imm(half)))
        for buf, st, sb in zip(bufs, s.state, s.state_b):
            ctx.bind(st, load_values(ctx, buf, mine, 1)[0])
            ctx.bind(sb, load_values(ctx, buf, other, 1)[0])
        lower_body(ctx, s.combine_states)
        for buf, st in zip(bufs, s.state):
            store_values(ctx, buf, mine, [ctx.lookup(st)])
        em.place(skip)
        _sync(ctx, s.barrier_id, s.barrier_count)
        half //= 2

    # Every thread reads the reduced value back into its own carried state.
    for buf, st in zip(bufs, s.state):
        v = load_values(ctx, buf, root, 1)[0]
        var = saved.get(st)
        if var is not None:
            ctx.mutable[st] = var
            ctx.move(var, v)
            ctx.bind(st, var)
        else:
            ctx.bind(st, v)

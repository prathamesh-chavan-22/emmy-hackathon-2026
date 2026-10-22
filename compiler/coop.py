"""Cross-thread statements: shared memory, barriers, warp shuffles, smem trees.

These are what turn per-thread partials into a block-wide result. They
register into the same dispatch table as the per-thread statements.
"""

from __future__ import annotations

import math

from . import expr as X
from .context import Buffer, Context, Value
from .stmts import _mem, address, flat_index, handles, load_values, lower_body, store_values

SMEM_DTYPES = {"float": "f32"}   # Smem carries the C type name
DYN_POOL = "dyn_smem"            # the extern pool behind dynamic shared memory


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

    When meta.json asks for dynamic shared memory (smem_bytes > 0, emmy's choice
    once the buffers outgrow the 48 KB static cap), every array is carved out of
    one extern pool at an aligned offset instead of being its own .shared symbol.
    """
    if s.name in ctx.buffers:
        raise ValueError(f"Smem {s.name!r} clashes with an existing buffer of that name")
    try:
        dt = SMEM_DTYPES[s.dtype]
    except KeyError:
        raise NotImplementedError(f"Smem dtype {s.dtype}") from None
    extents = tuple(int(e) for e in s.extents)
    nbytes = max(1, math.prod(extents)) * 4
    em = ctx.em
    base = em.reg("s32")
    if ctx.smem_pool:
        # Same packing as emmy's pack_smem (align to max(dtype size, align)), so
        # the layout fits exactly in the smem_bytes it computed for the launch.
        align = max(4, int(s.align))
        offset = -(-ctx.smem_used // align) * align
        ctx.smem_used = offset + nbytes
        ctx.smem_align = max(ctx.smem_align, align)   # the pool base must honour it too
        if ctx.smem_used > ctx.smem_pool:
            raise ValueError(f"shared arrays need {ctx.smem_used} B, launch provides {ctx.smem_pool} B")
        em.emit(f"mov.u32 \t{base}, {DYN_POOL}")
        if offset:
            em.emit(f"add.s32 \t{base}, {base}, {offset}", f"{s.name} at pool+{offset}")
    else:
        symbol = f"sh_{s.name}"
        em.declare_shared(symbol, nbytes, max(int(s.align), 16))   # 16: room for v4 access
        em.emit(f"mov.u32 \t{base}, {symbol}", s.name)
    ctx.buffers[s.name] = Buffer(s.name, base, dt, extents, space="shared")


@handles("CpAsyncCopy")
def lower_cp_async_copy(ctx: Context, s) -> None:
    """Asynchronous global -> shared copy of one 4/8/16-byte chunk, no register staging.

    16-byte copies use .cg (bypass L1, the streaming form); smaller ones .ca.
    `lane_index`, when set, is this thread's (row, col) inside a copy stripe and
    is added to the stripe's base element.
    """
    if s.swizzle not in ("NONE", "", None):
        raise NotImplementedError(f"cp.async swizzle {s.swizzle}")
    if s.nbytes not in (4, 8, 16):
        raise NotImplementedError(f"cp.async of {s.nbytes} bytes (PTX allows 4, 8, 16)")
    dst_buf, src_buf = ctx.buffers[s.smem], ctx.buffers[s.src]
    dst = flat_index(ctx, dst_buf, s.smem_index)
    if s.lane_index is not None:
        dst = X._binary(ctx, "+", dst, flat_index(ctx, dst_buf, s.lane_index))
    d_base, d_off = address(ctx, dst_buf, dst)
    s_base, s_off = address(ctx, src_buf, flat_index(ctx, src_buf, s.src_index))
    cache = "cg" if s.nbytes == 16 else "ca"
    ctx.em.emit(f"cp.async.{cache}.shared.global \t{_mem(d_base, d_off)}, {_mem(s_base, s_off)}, {s.nbytes}")


@handles("CpAsyncCommit")
def lower_cp_async_commit(ctx: Context, s) -> None:
    """Close this thread's outstanding cp.async copies into one group."""
    ctx.em.emit("cp.async.commit_group")


@handles("CpAsyncWait")
def lower_cp_async_wait(ctx: Context, s) -> None:
    """Wait until at most `group` committed groups are still in flight (pipelining depth)."""
    ctx.em.emit(f"cp.async.wait_group \t{int(s.group)}")


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

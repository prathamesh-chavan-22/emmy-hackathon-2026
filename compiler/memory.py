"""Addressing and memory access, shared by every statement that touches a buffer.

Global and shared buffers go through the same code: an element offset becomes
a (base register, constant byte offset) pair, and consecutive elements are
moved with v4/v2 vector instructions. Only the address width and the PTX state
space differ.
"""

from __future__ import annotations

from . import expr as X
from .context import Buffer, Context, Value

DTYPE_BYTES = {"f32": 4}   # every open problem is f32; others fail loudly, not silently


def dtype_bytes(dt: str) -> int:
    try:
        return DTYPE_BYTES[dt]
    except KeyError:
        raise NotImplementedError(f"dtype {dt}") from None


# An element offset is (dynamic part, constant part). Keeping the constant apart
# lets it ride the [reg+imm] address immediate instead of costing an add, which
# is what makes unrolled loops cheap: buf[i + 3] reuses buf[i]'s base register.
Offset = tuple[Value, int]


def at(v: Value) -> Offset:
    """A lowered element offset as an Offset."""
    return (X.imm(0), v.const) if v.is_imm else (v, 0)


def _affine(ctx: Context, e) -> Offset:
    """Lower an index expression, pulling constants out through + - and *k."""
    if type(e).__name__ == "BinaryExpr" and e.op in ("+", "-", "*"):
        (da, ca), (db, cb) = _affine(ctx, e.left), _affine(ctx, e.right)
        if e.op == "+":
            return X.binary(ctx, "+", da, db), ca + cb
        if e.op == "-":
            return X.binary(ctx, "-", da, db), ca - cb
        if db.is_imm:                 # (da + ca) * cb
            return X.binary(ctx, "*", da, X.imm(cb)), ca * cb
        if da.is_imm:                 # ca * (db + cb)
            return X.binary(ctx, "*", X.imm(ca), db), ca * cb
        # dynamic * dynamic: no constant to pull out
        whole = [X.binary(ctx, "+", d, X.imm(c)) for d, c in ((da, ca), (db, cb))]
        return X.binary(ctx, "*", *whole), 0
    if type(e).__name__ == "Var" and e.name in ctx.offsets:
        return ctx.offsets[e.name]
    return at(X.lower(ctx, e))


def flat_index(ctx: Context, buf: Buffer, index) -> Offset:
    """Row-major element offset of buf[index]; the IR gives one expression per dim."""
    dyn, const = X.imm(0), 0
    for e, stride in zip(index, buf.strides):
        d, c = _affine(ctx, e)
        dyn = X.binary(ctx, "+", dyn, X.binary(ctx, "*", d, X.imm(stride)))
        const += c * stride
    return dyn, const


def address(ctx: Context, buf: Buffer, flat: Offset) -> tuple[str, int]:
    """Address of element `flat` as (base register, constant byte offset).

    A constant offset goes into PTX's free [reg+imm] addressing, so unrolled
    cells share one base. Global addresses are 64-bit; shared ones are 32-bit
    offsets into the shared window.
    """
    em = ctx.em
    size = dtype_bytes(buf.dtype)
    dyn, const = flat
    if dyn.is_imm:
        return buf.addr, (dyn.const + const) * size
    # base + dyn * size in one multiply-add (mad.wide widens the index to 64-bit).
    if buf.space == "shared":
        addr = em.reg("s32")
        em.emit(f"mad.lo.s32 	{addr}, {dyn.text}, {size}, {buf.addr}")
        return addr, const * size
    addr = em.reg("u64")
    em.emit(f"mad.wide.s32 	{addr}, {dyn.text}, {size}, {buf.addr}")
    return addr, const * size


def operand(base: str, offset: int) -> str:
    """PTX memory operand: [base] or [base+offset]."""
    return f"[{base}+{offset}]" if offset else f"[{base}]"


def _vector_groups(n: int) -> list[tuple[int, int]]:
    """Split n consecutive elements into (start, width) chunks of v4/v2/scalar."""
    out, i = [], 0
    for w in (4, 2, 1):
        while n - i >= w:
            out.append((i, w))
            i += w
    return out


def load_values(ctx: Context, buf: Buffer, flat: Offset, count: int, names=()) -> list[Value]:
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
        em.emit(f"ld.{buf.space}{vec}.{dt} \t{dst}, {operand(base, off + start * size)}", note)
    return [Value(r, dt) for r in regs]


def store_values(ctx: Context, buf: Buffer, flat: Offset, values: list[Value], atomic=False, names=()) -> None:
    """Store consecutive elements: vector st, or one red.add per element when atomic."""
    em = ctx.em
    dt = buf.dtype
    base, off = address(ctx, buf, flat)
    vals = [X.to_reg(ctx, v).text for v in values]
    size = dtype_bytes(dt)
    if atomic:
        # No vector red.add for f32 before sm_90: one reduction per element.
        for i, v in enumerate(vals):
            em.emit(f"red.{buf.space}.add.{dt} \t{operand(base, off + i * size)}, {v}")
        return
    for start, w in _vector_groups(len(vals)):
        src = vals[start] if w == 1 else "{" + ", ".join(vals[start:start + w]) + "}"
        vec = "" if w == 1 else f".v{w}"
        note = f"{buf.name}[...] = {', '.join(names[start:start + w])}" if names else None
        em.emit(f"st.{buf.space}{vec}.{dt} \t{operand(base, off + start * size)}, {src}", note)

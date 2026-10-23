"""Addressing and memory access, shared by every statement that touches a buffer.

Global and shared buffers go through the same code: an element offset becomes
a (base register, constant byte offset) pair, and consecutive elements are
moved with v4/v2 vector instructions. Only the address width and the PTX state
space differ.

Elements are converted between their memory type and their compute kind here
(dtypes.py): an f16 load widens to f32, an f32 value stored to f16 rounds.
"""

from __future__ import annotations

from . import expr as X
from .context import Buffer, Context, Value
from .dtypes import cvt, get


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


# Smem swizzle modes: XOR mask over the 16-byte chunk index (bits 3.. of a half's
# index), taken from the index bits starting at `shift` (6 unless "MODE@shift").
SWIZZLE_MASKS = {"B128": 0x7, "B64": 0x3, "B32": 0x1}


def swizzle(ctx: Context, e: Value, mode: str) -> Value:
    """emmy's slab swizzle on element index e: e ^ (((e >> shift) & mask) << 3).

    It moves each 8-half (16-byte) chunk of a row to another chunk slot, chosen
    by the row, so the 8 rows one ldmatrix phase reads hit different banks.
    The cp.async fill and the ldmatrix drain apply the same permutation.
    """
    base, _, shift = mode.partition("@")
    if base in ("NONE", ""):
        return e
    if base not in SWIZZLE_MASKS:
        raise NotImplementedError(f"smem swizzle {mode}")
    mask, shift = SWIZZLE_MASKS[base], int(shift or 6)
    if e.is_imm:
        return X.imm(e.const ^ (((e.const >> shift) & mask) << 3))
    em = ctx.em
    field, r = em.reg("s32"), em.reg("s32")
    em.emit(f"bfe.u32 \t{field}, {e.text}, {shift}, {mask.bit_length()}", f"swizzle {mode}")
    em.emit(f"shl.b32 \t{field}, {field}, 3")
    em.emit(f"xor.b32 \t{r}, {e.text}, {field}")
    return Value(r, "s32")


def swizzled(ctx: Context, flat: Offset, mode) -> Offset:
    """flat through the slab swizzle `mode` (unchanged for NONE)."""
    if mode in ("NONE", "", None):
        return flat
    return swizzle(ctx, X.binary(ctx, "+", flat[0], X.imm(flat[1])), mode), 0


def address(ctx: Context, buf: Buffer, flat: Offset) -> tuple[str, int]:
    """Address of element `flat` as (base register, constant byte offset).

    A constant offset goes into PTX's free [reg+imm] addressing, so unrolled
    cells share one base. Global addresses are 64-bit; shared ones are 32-bit
    offsets into the shared window.
    """
    em = ctx.em
    size = get(buf.dtype).size
    dyn, const = flat
    if dyn.is_imm:
        return buf.addr, (dyn.const + const) * size
    # base + dyn * size in one multiply-add (mad.wide widens the index to 64-bit).
    if buf.space == "shared":
        addr = em.reg("s32")
        em.emit(f"mad.lo.s32 \t{addr}, {dyn.text}, {size}, {buf.addr}")
        return addr, const * size
    addr = em.reg("u64")
    em.emit(f"mad.wide.s32 \t{addr}, {dyn.text}, {size}, {buf.addr}")
    return addr, const * size


def _ptx_type(mem: str) -> str:
    """ld/st spell a half as raw .b16 bits; the cvt next to it gives them meaning."""
    return "b16" if mem == "f16" else mem


def operand(base: str, offset: int) -> str:
    """PTX memory operand: [base] or [base+offset]."""
    return f"[{base}+{offset}]" if offset else f"[{base}]"


def _vector_groups(n: int, size: int) -> list[tuple[int, int]]:
    """Split n consecutive elements into (start, width) chunks of v4/v2/scalar,
    at most 16 bytes per access (so f64 pairs at most)."""
    out, i = [], 0
    for w in (4, 2, 1):
        while n - i >= w and w * size <= 16:
            out.append((i, w))
            i += w
    return out


def load_values(ctx: Context, buf: Buffer, flat: Offset, count: int, names=()) -> list[Value]:
    """Load `count` consecutive elements starting at `flat`, vectorized (v4/v2) where possible."""
    em = ctx.em
    dt = get(buf.dtype)
    base, off = address(ctx, buf, flat)
    regs = [em.reg(dt.mem) for _ in range(count)]
    for start, w in _vector_groups(count, dt.size):
        dst = regs[start] if w == 1 else "{" + ", ".join(regs[start:start + w]) + "}"
        vec = "" if w == 1 else f".v{w}"
        note = f"{', '.join(names[start:start + w])} = {buf.name}[...]" if names else None
        em.emit(f"ld.{buf.space}{vec}.{_ptx_type(dt.mem)} \t{dst}, {operand(base, off + start * dt.size)}", note)
    if dt.mem == dt.kind:
        return [Value(r, dt.kind) for r in regs]
    out = [em.reg(dt.kind) for _ in regs]
    for r, o in zip(regs, out):
        cvt(em, o, dt.kind, r, dt.mem)
    return [Value(o, dt.kind) for o in out]


def store_values(ctx: Context, buf: Buffer, flat: Offset, values: list[Value], atomic=False, names=()) -> None:
    """Store consecutive elements: vector st, or one red.add per element when atomic."""
    em = ctx.em
    dt = get(buf.dtype)
    base, off = address(ctx, buf, flat)
    # Each value as the buffer's element (a bool stored to a float buffer is
    # 1.0/0.0 and a float stored to an int one truncates, as in emmy's C).
    vals = [X.to_reg(ctx, X.convert(ctx, v, dt.kind)).text for v in values]
    if dt.mem != dt.kind:
        narrowed = [em.reg(dt.mem) for _ in vals]
        for n, v in zip(narrowed, vals):
            cvt(em, n, dt.mem, v, dt.kind)
        vals = narrowed
    if atomic:
        # No vector red.add before sm_90: one reduction per element.
        for i, v in enumerate(vals):
            em.emit(f"red.{buf.space}.add.{dt.atomic} \t{operand(base, off + i * dt.size)}, {v}")
        return
    for start, w in _vector_groups(len(vals), dt.size):
        src = vals[start] if w == 1 else "{" + ", ".join(vals[start:start + w]) + "}"
        vec = "" if w == 1 else f".v{w}"
        note = f"{buf.name}[...] = {', '.join(names[start:start + w])}" if names else None
        em.emit(f"st.{buf.space}{vec}.{_ptx_type(dt.mem)} \t{operand(base, off + start * dt.size)}, {src}", note)

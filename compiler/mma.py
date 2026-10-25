"""Tensor-core statements: mma.sync m16n8k16 (f16 x f16 -> f32) and its register fragments.

emmy lowers an f16 matmul (and attention's two matmuls) to warp-wide tensor
core tiles. Each lane of the warp holds a slice of every operand tile in
registers - a *fragment* - laid out the way the PTX `mma.sync.m16n8k16` spec
fixes it:

- A (16x16, f16): 4 x .b32 per lane, two halves each.
- B (16x8, f16): 2 x .b32 per lane.
- C (16x8, f32): 4 x .f32 per lane. Lane l owns rows g = l/4 and g+8, columns
  2t and 2t+1 with t = l%4: elements [0,1] are row g, [2,3] row g+8.

A fragment is a list of registers updated in place (the mma accumulates into
its C registers), so it lives across loop iterations like any carried value.
emmy groups sibling fragments into families (`_c0_[2][4]`) and picks a member
with a loop variable (`_c0_[_r0]`); those loops are unrolled (stmts.py), so the
member index is always a compile-time constant here.

Operands reach the fragments two ways: from shared memory with `ldmatrix`
(the slab was filled by cp.async, possibly through an XOR swizzle that spreads
rows over the memory banks), or straight from global memory, one half at a
time, following the same lane-to-element map.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import expr as X
from .context import Context, Value
from .dtypes import bits, get
from .memory import address, flat_index, load_values, operand, store_values, swizzle, swizzled
from .stmts import compute, handles, lower_body

# FragmentApply argument kinds (emmy's FRAG / ROW / UNIFORM / COORD / GMEM).
FRAG, ROW, UNIFORM, COORD, GMEM = "frag", "row", "uniform", "coord", "gmem"
# Reserved IR names: an element's absolute (row, col) in a FragmentApply COORD or
# GMEM template, and its (row, col) offset inside the tile in a RegStore epilogue.
FRAG_ROW, FRAG_COL = "__frow", "__fcol"
ELEM_ROW, ELEM_COL = "__M__", "__N__"

ELEM_ROWS = (0, 0, 1, 1)       # C element -> which of the lane's two rows (g, g+8)


@dataclass
class Fragment:
    """A family of `count` register arrays (count 1 for a plain fragment)."""
    members: list[list[str]]


def frag(ctx: Context, ref: str) -> list[str]:
    """The registers of fragment reference `name` or `name[i]` (i a constant or a bound name)."""
    stem, _, rest = ref.partition("[")
    family = ctx.frags.get(stem)
    if family is None:
        raise KeyError(f"fragment {ref!r} used before its RegFragment")
    if not rest:
        return family.members[0]
    idx = rest.rstrip("]")
    i = int(idx) if idx.isdigit() else ctx.lookup(idx)
    if isinstance(i, Value):
        if not i.is_imm:
            raise NotImplementedError(f"fragment member {ref!r} picked by a runtime value")
        i = i.const
    return family.members[i]


def _new_frag(ctx: Context, name: str, kind: str, n: int, count: int = 1) -> Fragment:
    f = Fragment([[ctx.em.reg(kind) for _ in range(n)] for _ in range(count)])
    ctx.frags[name] = f
    return f


# ---------------------------------------------------------------------------
# Lane geometry
# ---------------------------------------------------------------------------

def _lane_parts(ctx: Context) -> tuple[Value, Value, Value]:
    """(lane, g, t) with lane = tid & 31, g = lane >> 2, t = lane & 3.

    Recomputed at each use (three cheap ops ptxas merges), so no statement
    depends on a register defined on some other control path.
    """
    em = ctx.em
    lane, g, t = em.reg("s32"), em.reg("s32"), em.reg("s32")
    em.emit(f"and.b32 \t{lane}, {ctx.lookup('%tid').text}, 31", "lane")
    em.emit(f"shr.u32 \t{g}, {lane}, 2", "g = lane / 4")
    em.emit(f"and.b32 \t{t}, {lane}, 3", "t = lane % 4")
    return Value(lane, "s32"), Value(g, "s32"), Value(t, "s32")


def _bitop(ctx: Context, instr: str, a: Value, k: int) -> Value:
    r = ctx.em.reg("s32")
    ctx.em.emit(f"{instr} \t{r}, {a.text}, {k}")
    return Value(r, "s32")


def _whole(ctx: Context, flat) -> Value:
    """An Offset (register part, constant part) as one value (the swizzle needs all of it)."""
    return X.binary(ctx, "+", flat[0], X.imm(flat[1]))


def _min(ctx: Context, a: Value, b: Value) -> Value:
    r = ctx.em.reg("s32")
    ctx.em.emit(f"min.s32 \t{r}, {X.to_reg(ctx, a).text}, {b.text}")
    return Value(r, "s32")


# ---------------------------------------------------------------------------
# Declarations and the mma itself
# ---------------------------------------------------------------------------

@handles("RegFragment")
def lower_reg_fragment(ctx: Context, s) -> None:
    """Declare a fragment (family): fresh registers; an f32 accumulator starts at 0.

    Register counts are the m16n8k16 ones (see the module docstring). Only
    f16 operands with an f32 accumulator are lowered: bf16 and fp8 inputs
    cannot be fed to this runner, and emmy emits an f16 accumulator only
    behind an opt-in fast-math flag. Other shapes or types are rejected.
    """
    if tuple(s.shape) != (16, 8, 16):
        raise NotImplementedError(f"mma fragment shape {s.shape}")
    dtype = getattr(s.dtype, "name", str(s.dtype))
    count = int(s.count)
    if s.role in ("a", "b") and dtype == "f16":
        _new_frag(ctx, s.name, "b32", 4 if s.role == "a" else 2, count)
    elif s.role == "c" and dtype == "f32":
        f = _new_frag(ctx, s.name, "f32", 4, count)
        for regs in f.members:
            for r in regs:
                ctx.em.emit(f"mov.f32 \t{r}, 0f00000000")
    else:
        raise NotImplementedError(f"mma fragment role {s.role} dtype {dtype}")


@handles("MmaSyncPtx")
def lower_mma_sync(ctx: Context, s) -> None:
    """c += a @ b for one 16x8 tile over k = 16: one warp-wide tensor-core instruction.

    Inputs are rounded f16, products and sums are f32 (the .f32 accumulator form).
    """
    if tuple(s.shape) != (16, 8, 16) or s.ab_dtype != "f16" or s.c_dtype != "f32" or s.b_row_major:
        raise NotImplementedError(f"mma.sync {s.shape} {s.ab_dtype}->{s.c_dtype}")
    c, a, b = frag(ctx, s.c_frag), frag(ctx, s.a_frag), frag(ctx, s.b_frag)
    group = lambda regs: "{" + ", ".join(regs) + "}"
    ctx.em.emit(f"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \t"
                f"{group(c)}, {group(a)}, {group(b)}, {group(c)}", f"{s.c_frag} += {s.a_frag} @ {s.b_frag}")


# ---------------------------------------------------------------------------
# Operand loads
# ---------------------------------------------------------------------------

@handles("LdmatrixLoad")
def lower_ldmatrix_load(ctx: Context, s) -> None:
    """Fill an A or B fragment from its tile, from shared (ldmatrix) or global memory."""
    if s.fragment_layout != "m16n8k16" or s.byte_slab or s.scale_buffer is not None:
        raise NotImplementedError(f"LdmatrixLoad layout {s.fragment_layout}")
    buf = ctx.buffers[s.src_buffer]
    ldm = int(s.ldm) or buf.shape[-1]
    if s.staged:
        _ldmatrix(ctx, s, buf, ldm)
    else:
        _gmem_fragment(ctx, s, buf, ldm)


def _ldmatrix(ctx: Context, s, buf, ldm: int) -> None:
    """ldmatrix: the warp reads 8x8 half matrices from smem; lane L supplies one row address.

    Which rows each lane points at (and .trans for a row-major B slab) is what
    makes the registers land in the mma layout:
    - A, x4: four 8x8 blocks of the 16x16 tile; lane L -> row L%16, col block L/16.
    - B, x2.trans: the 16x8 tile is K-major, so rows L%16 are transposed on load.
    - B stored N-major (b_trans), plain x2: rows L%8, k block (L/8)%2.
    - pair: one x4 fills two B fragments (columns n and n+8) at once.
    """
    if get(buf.dtype).mem != "f16":
        raise NotImplementedError(f"ldmatrix from a {buf.dtype} slab")
    lane = _lane_parts(ctx)[0]
    lo16, hi = _bitop(ctx, "and.b32", lane, 15), _bitop(ctx, "shr.u32", lane, 4)
    if s.pair_frag is not None:
        if s.b_trans:   # rows (L%8) + (L/16)*8, k block (L/8)%2
            row = X.binary(ctx, "+", _bitop(ctx, "and.b32", lane, 7), X.binary(ctx, "*", hi, X.imm(8)))
            col = X.binary(ctx, "*", _bitop(ctx, "and.b32", _bitop(ctx, "shr.u32", lane, 3), 1), X.imm(8))
            instr = "x4"
        else:
            row, col, instr = lo16, X.binary(ctx, "*", hi, X.imm(8)), "x4.trans"
        regs = frag(ctx, s.frag)[:2] + frag(ctx, s.pair_frag)[:2]
    elif s.role == "a":
        row, col, instr = lo16, X.binary(ctx, "*", hi, X.imm(8)), "x4"
        regs = frag(ctx, s.frag)
    elif s.b_trans:
        row = _bitop(ctx, "and.b32", lane, 7)
        col = X.binary(ctx, "*", _bitop(ctx, "and.b32", _bitop(ctx, "shr.u32", lane, 3), 1), X.imm(8))
        instr, regs = "x2", frag(ctx, s.frag)
    else:
        row, col, instr, regs = lo16, X.imm(0), "x2.trans", frag(ctx, s.frag)
    mine = X.binary(ctx, "+", X.binary(ctx, "*", row, X.imm(ldm)), col)
    elem = swizzle(ctx, X.binary(ctx, "+", _whole(ctx, flat_index(ctx, buf, s.src_index)), mine), s.swizzle)
    base, off = address(ctx, buf, (elem, 0))
    ctx.em.emit(f"ldmatrix.sync.aligned.m8n8.{instr}.shared.b16 \t{{{', '.join(regs)}}}, {operand(base, off)}",
                f"{s.frag}{'' if s.pair_frag is None else ', ' + s.pair_frag} <- {buf.name}")


def _gmem_fragment(ctx: Context, s, buf, ldm: int) -> None:
    """A fragment read straight from global memory (the operand was not staged in smem).

    Each register packs two halves (low = first). Coordinates within the tile:
    - A reg i: row g + 8*(i&1), k = 2t + 8*(i>>1) and k+1        -> buf[row][k]
    - B reg i: col n = g, k = 2t + 8*i and k+1                   -> buf[k][n], or
      buf[n][k] when B is stored N-major (b_trans).
    gmem_guard clamps the lane's row (A) or n (B) into the tile's live range,
    min(x, max(left-1, 0)): a duplicate in-bounds read whose result the
    store guard drops. k_zero zero-fills k past the bound instead: k is
    summed by the mma, so it must contribute exactly 0.
    """
    em = ctx.em
    _, g, t = _lane_parts(ctx)
    t2 = _bitop(ctx, "shl.b32", t, 1)
    clamp = k_left = None
    if s.gmem_guard is not None:
        tile_base, tile_bound = (X.lower(ctx, e) for e in s.gmem_guard)
        left = X.binary(ctx, "-", X.binary(ctx, "-", tile_bound, tile_base), X.imm(1))
        clamp = X.imm(max(left.const, 0)) if left.is_imm else Value(em.reg("s32"), "s32")
        if not left.is_imm:
            em.emit(f"max.s32 \t{clamp.text}, {left.text}, 0", "clamp row/col to the live range")
    if s.k_zero is not None:
        k_base, k_bound = (X.lower(ctx, e) for e in s.k_zero)
        k_left = X.binary(ctx, "-", k_bound, k_base)
    dyn = flat_index(ctx, buf, s.src_index)
    pick = lambda x: x if clamp is None else _min(ctx, x, clamp)

    # (row, col) of each half in buf, as (register, constant) pairs, and its k.
    halves = []
    if s.role == "a":
        rows = [pick(g), pick(X.binary(ctx, "+", g, X.imm(8)))] if clamp is not None else [g, g]
        for i in range(4):
            row, row_c = (rows[i & 1], 0) if clamp is not None else (g, 8 * (i & 1))
            k0 = 8 * (i >> 1)
            halves.append([((row, row_c), (t2, k0 + j), k0 + j) for j in (0, 1)])
    else:
        n = pick(g)
        for i in range(2):
            k0 = 8 * i
            if s.b_trans:
                halves.append([((n, 0), (t2, k0 + j), k0 + j) for j in (0, 1)])
            else:
                halves.append([((t2, k0 + j), (n, 0), k0 + j) for j in (0, 1)])

    memo = {}
    dt = get(buf.dtype)
    regs = frag(ctx, s.frag)
    for reg, pair in zip(regs, halves):
        packed = []
        for (row, row_c), (col, col_c), k in pair:
            key = (row.text, col.text)
            if key not in memo:   # one base address per (row, col) register pair
                memo[key] = address(ctx, buf, (X.binary(ctx, "+", dyn[0], X.binary(
                    ctx, "+", X.binary(ctx, "*", row, X.imm(ldm)), col)), dyn[1]))
            addr, off = memo[key]
            off += (row_c * ldm + col_c) * dt.size
            raw = em.reg(dt.mem)
            ld = f"ld.global.{'b16' if dt.mem == 'f16' else dt.mem} \t{raw}, {operand(addr, off)}"
            if k_left is None:
                em.emit(ld)
            else:   # in range when 2t + k < k_left
                p = X.binary(ctx, "<", X.binary(ctx, "+", t2, X.imm(k)), k_left)  # t2 is a register
                em.emit(f"mov.b{bits(dt.mem)} \t{raw}, 0", "k past the bound reads 0")
                em.emit(f"@{p.text} {ld}")
            if dt.mem != "f16":
                h = em.reg("f16")
                em.emit(f"cvt.rn.f16.{dt.mem} \t{h}, {raw}")
                raw = h
            packed.append(raw)
        em.emit(f"mov.b32 \t{reg}, {{{packed[0]}, {packed[1]}}}")


# ---------------------------------------------------------------------------
# Accumulator store
# ---------------------------------------------------------------------------

@handles("RegStore")
def lower_reg_store(ctx: Context, s) -> None:
    """Write a C fragment to its buffer: lane's elements at rows g/g+8, cols 2t/2t+1.

    The two columns of a row are adjacent, so each row is one v2 store. An
    epilogue (a fused bias, residual or activation) is evaluated per element
    right before its store, with the element's tile offsets bound to emmy's
    reserved names. m_guard/n_guard mask the elements of a tile that hangs
    over the buffer edge; an n_guard splits the pairs, since the two columns
    can fall on different sides of the edge. The epilogue's own loads sit
    under the same guard, as they read the same coordinates.
    """
    if s.fragment_layout != "m16n8k16" or s.volta_interleaved:
        raise NotImplementedError(f"RegStore layout {s.fragment_layout}")
    em = ctx.em
    buf = ctx.buffers[s.dst_buffer]
    strides = buf.strides
    ldm = int(s.ldm) or (strides[s.row_dim] if s.row_dim is not None else buf.shape[-1])
    ldn = strides[s.col_dim] if s.col_dim is not None else 1
    frags = [frag(ctx, f) for f in (s.frag, *s.extra_frags)]
    _, g, t = _lane_parts(ctx)
    t2 = _bitop(ctx, "shl.b32", t, 1)
    rows = (g, X.binary(ctx, "+", g, X.imm(8)))
    cols = (t2, X.binary(ctx, "+", t2, X.imm(1)))
    coords = [(rows[ELEM_ROWS[i]], cols[i & 1]) for i in range(4)]

    def guard(i: int):
        preds = []
        for g_, off in ((s.m_guard, coords[i][0]), (s.n_guard, coords[i][1])):
            if g_ is not None:
                base, bound = (X.lower(ctx, e) for e in g_)
                preds.append(X.binary(ctx, "<", X.binary(ctx, "+", base, off), bound))
        p = X.imm(True)
        for q in preds:
            p = X.binary(ctx, "&&", p, q) if not (p.is_imm and p.const) else q
        return p

    def value(i: int) -> Value:
        if s.epilogue is None:
            return Value(frags[0][i], "f32")
        epi = s.epilogue
        with ctx.scope():
            for param, f in zip(epi.params, frags):
                ctx.bind(param, Value(f[i], "f32"))
            ctx.bind(ELEM_ROW, coords[i][0])
            ctx.bind(ELEM_COL, coords[i][1])
            lower_body(ctx, epi.body)
            return ctx.lookup(epi.results[0])

    dyn, const = flat_index(ctx, buf, s.dst_index)
    paired = s.n_guard is None and ldn == 1 and not s.atomic
    groups = [(0, 1), (2, 3)] if paired else [(i,) for i in range(4)]
    for group in groups:
        p = guard(group[0])
        if p.is_imm and not p.const:
            continue
        skip = None
        if not p.is_imm:
            skip = em.label("frag_skip")
            em.emit(f"@!{p.text} bra \t{skip}", "element outside the buffer")
        with ctx.scope():
            values = [value(i) for i in group]
            row, col = coords[group[0]]
            elem = X.binary(ctx, "+", X.binary(ctx, "*", row, X.imm(ldm)), X.binary(ctx, "*", col, X.imm(ldn)))
            # Swizzle is a shared-slab layout; atomics add into global memory (split-K), as in emmy.
            flat = swizzled(ctx, (X.binary(ctx, "+", dyn, elem), const), "NONE" if s.atomic else s.swizzle)
            store_values(ctx, buf, flat, values, s.atomic)
        if skip:
            em.place(skip)


# ---------------------------------------------------------------------------
# Elementwise work on C fragments (attention's softmax between its two matmuls)
# ---------------------------------------------------------------------------

def _apply(ctx: Context, op: str, values: list[Value]) -> Value:
    """op(values) in f32, through the scalar Assign lowering."""
    with ctx.scope():
        names = [f"__arg{k}" for k in range(len(values))]
        for n, v in zip(names, values):
            ctx.bind(n, v)
        return compute(ctx, op, names, None)


@handles("FragmentApply")
def lower_fragment_apply(ctx: Context, s) -> None:
    """out[e] = post(op(args at element e)) for each of the 4 C elements.

    A FRAG arg is that element of another fragment, a ROW arg the lane's value
    for the element's row (g or g+8), a UNIFORM arg one scalar, a COORD arg a
    predicate over the element's absolute (row, col), and a GMEM arg a load
    at those coordinates (an additive attention mask, say). Results go to
    fresh registers, then into `out` - in place, out may also be an input.
    """
    if s.layout.n_elems != 4 or tuple(s.layout.elem_row) != ELEM_ROWS:
        raise NotImplementedError("FragmentApply on a non-m16n8 layout")
    em = ctx.em
    need_coords = any(k in (COORD, GMEM) for k in s.kinds)
    coords = None
    if need_coords:
        _, g, t = _lane_parts(ctx)
        t2 = _bitop(ctx, "shl.b32", t, 1)
        rows = (g, X.binary(ctx, "+", g, X.imm(8)))
        cols = (t2, X.binary(ctx, "+", t2, X.imm(1)))
        coords = [(rows[ELEM_ROWS[i]], cols[i & 1]) for i in range(4)]
    results = []
    for i in range(4):
        with ctx.scope():
            if coords is not None:
                if s.row_base is not None:
                    ctx.bind(FRAG_ROW, X.binary(ctx, "+", X.lower(ctx, s.row_base), coords[i][0]))
                if s.col_base is not None:
                    ctx.bind(FRAG_COL, X.binary(ctx, "+", X.lower(ctx, s.col_base), coords[i][1]))
            args = []
            for a, kind in zip(s.args, s.kinds):
                if kind == FRAG:
                    args.append(Value(frag(ctx, a)[i], "f32"))
                elif kind == ROW:
                    args.append(ctx.lookup(a[ELEM_ROWS[i]]))
                elif kind == UNIFORM:
                    args.append(ctx.lookup(a) if isinstance(a, str) else X.lower(ctx, a))
                elif kind == COORD:
                    args.append(X.lower(ctx, a))
                elif kind == GMEM:
                    buf = ctx.buffers[a[0]]
                    args.append(load_values(ctx, buf, flat_index(ctx, buf, a[1]), 1)[0])
                else:
                    raise NotImplementedError(f"FragmentApply arg kind {kind}")
            v = _apply(ctx, s.op.name, args)
            for post in s.post:
                v = _apply(ctx, post.name, [v])
            results.append(X.to_reg(ctx, X.convert(ctx, v, "f32")))
    if s.in_place or s.out in ctx.frags or "[" in s.out:
        out = frag(ctx, s.out)
    else:
        out = _new_frag(ctx, s.out, "f32", 4).members[0]
    for r, v in zip(out, results):
        if r != v.text:
            em.emit(f"mov.f32 \t{r}, {v.text}")


@handles("FragmentRowReduce")
def lower_fragment_row_reduce(ctx: Context, s) -> None:
    """Per-row max or sum over a chunk of C fragments, then across the row's 4 lanes.

    top folds every element in row g (elements 0, 1 of each fragment, in
    order), bot those in row g+8. A row's columns are spread over the 4 lanes
    with the same g, which differ in the low two lane bits: a butterfly
    shuffle over masks 2 and 1 gives each of them the whole row's result.
    The member mask is the full warp (-1): mma.sync itself needs all 32 lanes,
    so unlike a plain reduction (coop.py) no partial warp can get here.
    """
    if tuple(s.layout.reduce_xors) != (2, 1) or tuple(s.layout.elem_row) != ELEM_ROWS:
        raise NotImplementedError("FragmentRowReduce on a non-m16n8 layout")
    em = ctx.em
    op = s.op.name
    regs = [frag(ctx, f) for f in s.frags]
    for name, row in ((s.top, 0), (s.bot, 1)):
        parts = [Value(f[i], "f32") for f in regs for i in range(4) if ELEM_ROWS[i] == row]
        acc = parts[0]
        for p in parts[1:]:
            acc = _apply(ctx, op, [acc, p])
        for mask in s.layout.reduce_xors:
            other = em.reg("f32")
            em.emit(f"shfl.sync.bfly.b32 \t{other}, {X.to_reg(ctx, acc).text}, {mask}, 31, -1",
                    f"{name}: lane ^ {mask}")
            acc = _apply(ctx, op, [acc, Value(other, "f32")])
        ctx.assign(name, acc)


@handles("FragmentRepack")
def lower_fragment_repack(ctx: Context, s) -> None:
    """Two adjacent C fragments (f32, 16x8 each) -> one A fragment (f16, 16x16), in registers.

    The C layout of columns 0..7 and 8..15 matches A's layout of k 0..7 and
    8..15 lane for lane, so each A register is just two C elements rounded to
    f16 and packed (cvt.rn.f16x2 puts its first source in the high half).
    """
    if s.fragment_layout != "m16n8k16" or s.ab_dtype != "f16" or len(s.srcs) != 2:
        raise NotImplementedError(f"FragmentRepack {s.fragment_layout} {s.ab_dtype}")
    a = frag(ctx, s.frag)
    c0, c1 = (frag(ctx, f) for f in s.srcs)
    for dst, (hi, lo) in zip(a, ((c0[1], c0[0]), (c0[3], c0[2]), (c1[1], c1[0]), (c1[3], c1[2]))):
        ctx.em.emit(f"cvt.rn.f16x2.f32 \t{dst}, {hi}, {lo}")

"""Element types: how each IR dtype is stored, loaded and computed in.

emmy's C keeps every variable at its declared type but does almost all math
in float. This table mirrors that, so the rest of the compiler only asks
"what kind does this value compute in" and "how does it move to memory".
Where emmy's C and torch disagree, we side with torch: the expected outputs
are torch's.

- f32 is the native compute kind.
- f16 values live in f32 registers. Every f16-typed result is rounded to half
  precision on assignment, which is what a `__half` variable does. For + - * /
  and sqrt this is bit-identical to native half math: f32 carries more than
  twice f16's precision, so rounding twice never differs from rounding once.
- bf16 and fp8 (e4m3, e5m2) work the same way: f32 registers, rounded to the
  narrow format on assignment. The runner cannot allocate them, so they only
  occur inside a kernel (`x.to(torch.bfloat16)`, a bf16 tensor-core operand
  slab). fp8 follows torch at the edge of the range: e4m3fn has no infinity,
  so torch saturates it (any overflow, even inf, is +-448); e5m2 overflows to inf.
- f64 computes natively for the correctly rounded ops (+ - * / sqrt, min,
  max, ...), so those match torch exactly; emmy narrows f64 to f32 for all
  math, which cannot. Transcendentals go through f32, as in emmy.
- i32/i64 compute natively for the integer ops emmy renders as C operators
  (add, mul, //, %, shifts, bitwise), with torch's floor semantics for // and
  %; everything else goes through f32 and is truncated back, as C's implicit
  conversions do.

A value's register kind is its PTX type name, so conversions are one `cvt`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .emitter import f32_imm


@dataclass(frozen=True)
class DType:
    name: str        # IR / meta.json name
    size: int        # bytes per element in memory
    mem: str         # PTX type of ld/st/red, and of the register a load lands in
    kind: str        # register kind its values compute in
    atomic: str      # PTX type of red.add

    @property
    def rounds(self) -> bool:
        """Values compute in a wider kind and are rounded to the storage type on assignment."""
        return self.mem in NARROW


@dataclass(frozen=True)
class Narrow:
    """A float format narrower than f32: its precision (for constants) and overflow rule."""
    mantissa: int      # stored mantissa bits
    min_exp: int       # exponent of the smallest normal
    max: float         # largest finite value
    overflow: float    # what torch makes of a value that rounds past max


NARROW = {
    "f16": Narrow(10, -14, 65504.0, math.inf),
    "bf16": Narrow(7, -126, 3.3895313892515355e38, math.inf),
    "e4m3": Narrow(3, -6, 448.0, 448.0),
    "e5m2": Narrow(2, -14, 57344.0, math.inf),
}

DTYPES = {d.name: d for d in (
    DType("f32", 4, "f32", "f32", "f32"),
    DType("f16", 2, "f16", "f32", "noftz.f16"),
    DType("f64", 8, "f64", "f64", "f64"),
    DType("i32", 4, "s32", "s32", "s32"),
    DType("i64", 8, "s64", "s64", "u64"),   # two's complement: unsigned add is the same bits
    DType("bf16", 2, "bf16", "f32", "noftz.bf16"),
    DType("f8e4m3", 1, "e4m3", "f32", "e4m3"),
    DType("f8e5m2", 1, "e5m2", "f32", "e5m2"),
)}

# C type names emmy uses for shared-memory arrays.
C_TYPES = {"float": "f32", "__half": "f16", "double": "f64", "int": "i32", "long long": "i64",
           "__nv_bfloat16": "bf16", "__nv_fp8_e4m3": "f8e4m3", "__nv_fp8_e5m2": "f8e5m2"}

INT_KINDS = ("s32", "s64")
FLOAT_KINDS = ("f16", "f32", "f64")


def get(name: str) -> DType:
    try:
        return DTYPES[name]
    except KeyError:
        raise NotImplementedError(f"dtype {name} (supported: {', '.join(DTYPES)})") from None


def bits(kind: str) -> int:
    return (16 if kind in ("f16", "bf16", "b16") else 8 if kind in ("e4m3", "e5m2")
            else 64 if kind in ("s64", "f64") else 32)


def common_kind(kinds) -> str:
    """The kind C's usual arithmetic conversions pick: double, else float, else the widest int."""
    kinds = set(kinds) - {"pred"}
    if "f64" in kinds:
        return "f64"
    if kinds & set(FLOAT_KINDS) or not kinds:
        return "f32"
    return "s64" if "s64" in kinds else "s32"


def round_const(value: float, fmt: Narrow) -> float:
    """value rounded to nearest-even in a narrow float format, as round_to does on the GPU."""
    if value == 0 or not math.isfinite(value):
        return value
    exp = max(math.floor(math.log2(abs(value))), fmt.min_exp)
    step = 2.0 ** (exp - fmt.mantissa)
    r = round(value / step) * step        # Python's round() is half-to-even
    return math.copysign(fmt.overflow, value) if abs(r) > fmt.max else r


def cast_const(value, kind: str, rounds: str | None = None):
    """A compile-time constant converted as the hardware cvt would.

    float -> int truncates and saturates (cvt.rzi; NaN -> 0), which is what
    emmy's `int acc = -1e30f` max identity becomes on the GPU. `rounds` names
    the narrow format (NARROW) a float constant is rounded to first.
    """
    if kind == "pred":
        return bool(value)
    if kind in INT_KINDS:
        if isinstance(value, float):
            if math.isnan(value):
                return 0
            lo, hi = (-2**31, 2**31 - 1) if kind == "s32" else (-2**63, 2**63 - 1)
            return int(min(max(value, lo), hi)) if math.isfinite(value) else (hi if value > 0 else lo)
        width = bits(kind)
        v = int(value) & ((1 << width) - 1)
        return v - (1 << width) if v >> (width - 1) else v
    if kind == "f64":
        return float(value)
    with np.errstate(over="ignore"):
        v = float(np.float32(value))
        if rounds == "f16":
            return float(np.float16(v))
        return round_const(v, NARROW[rounds]) if rounds else v


def cvt(em, dst: str, dst_kind: str, src: str, src_kind: str) -> None:
    """dst = src converted between numeric kinds, with C's rounding rules:
    float -> int truncates, int -> float and float narrowing round to nearest."""
    if src_kind == dst_kind:
        em.emit(f"mov.b{bits(dst_kind)} \t{dst}, {src}")
        return
    if (dst_kind, src_kind) == ("bf16", "f32"):
        em.emit(f"cvt.rn.bf16.f32 \t{dst}, {src}")
        return
    if (dst_kind, src_kind) == ("f32", "bf16"):
        _bf16_to_f32(em, dst, src)
        return
    if "bf16" in (dst_kind, src_kind) or {dst_kind, src_kind} & {"e4m3", "e5m2"}:
        raise NotImplementedError(f"conversion {src_kind} -> {dst_kind}")
    if dst_kind in INT_KINDS and src_kind in INT_KINDS:
        mode = ""
    elif dst_kind in INT_KINDS:
        mode = ".rzi"
    elif src_kind in INT_KINDS or bits(dst_kind) < bits(src_kind):
        mode = ".rn"
    else:
        mode = ""     # widening float: exact
    em.emit(f"cvt{mode}.{dst_kind}.{src_kind} \t{dst}, {src}")


def _bf16_to_f32(em, dst: str, src: str) -> None:
    """bf16 is the top half of an f32, so widening is a shift (cvt.f32.bf16 needs sm_90)."""
    r = em.reg("b32")
    em.emit(f"cvt.u32.u16 \t{r}, {src}")
    em.emit(f"shl.b32 \t{r}, {r}, 16")
    em.emit(f"mov.b32 \t{dst}, {r}")


def round_to(em, value: str, dt: DType) -> str:
    """An f32 register rounded to dt's precision (a new register), when dt rounds.

    Each format goes through the hardware conversion and back:
    - f16: cvt.rn.f16.f32, cvt.f32.f16.
    - bf16: cvt.rn.bf16.f32, then the 16 bits shifted back into an f32.
    - fp8 has no usable cvt: the e4m3x2/e5m2x2 forms need PTX 8.1 on sm_89 and
      the target is fixed at 8.0. So it is rounded in f32 arithmetic: with
      C = 2^(e + 23 - m) for |x| in [2^e, 2^(e+1)) (m mantissa bits, e no lower
      than the smallest normal exponent), |x| + C lands in [C, 2C) where the
      f32 step is exactly fp8's step at x, so add.rn rounds to nearest-even
      there, and subtracting C again is exact. Overflow is selected
      separately, as torch has it: +-448 for e4m3fn, infinity for e5m2, from
      the midpoint between max and the next step.
    """
    if not dt.rounds:
        return value
    f = em.reg("f32")
    if dt.mem == "f16":
        h = em.reg("f16")
        em.emit(f"cvt.rn.f16.f32 \t{h}, {value}")
        em.emit(f"cvt.f32.f16 \t{f}, {h}", f"round to {dt.name}")
        return f
    if dt.mem == "bf16":
        h = em.reg("bf16")
        em.emit(f"cvt.rn.bf16.f32 \t{h}, {value}")
        _bf16_to_f32(em, f, h)
        return f
    fmt = NARROW[dt.mem]
    mag, bin_, c, t = em.reg("f32"), em.reg("b32"), em.reg("f32"), em.reg("f32")
    em.emit(f"abs.f32 \t{mag}, {value}")
    em.emit(f"mov.b32 \t{bin_}, {mag}")
    em.emit(f"and.b32 \t{bin_}, {bin_}, 0x7F800000", "2^e: the exponent field alone")
    em.emit(f"mov.b32 \t{c}, {bin_}")
    em.emit(f"mul.rn.f32 \t{c}, {c}, {f32_imm(2.0 ** (23 - fmt.mantissa))}")
    em.emit(f"max.f32 \t{c}, {c}, {f32_imm(2.0 ** (fmt.min_exp + 23 - fmt.mantissa))}", "subnormal step")
    em.emit(f"add.rn.f32 \t{t}, {mag}, {c}", "rounds at the format's step")
    em.emit(f"sub.rn.f32 \t{t}, {t}, {c}")
    em.emit(f"copysign.f32 \t{f}, {value}, {t}", f"round to {dt.name}")
    # A tie at the midpoint rounds to even: max's mantissa is odd for e5m2 (so
    # the tie overflows) and even for e4m3 (so the tie stays at max).
    edge = fmt.max + 2.0 ** (math.floor(math.log2(fmt.max)) - fmt.mantissa) / 2
    over, out, big = em.reg("f32"), em.reg("f32"), em.reg("pred")
    em.emit(f"setp.{'ge' if math.isinf(fmt.overflow) else 'gt'}.f32 \t{big}, {mag}, {f32_imm(edge)}")
    em.emit(f"copysign.f32 \t{over}, {value}, {f32_imm(fmt.overflow)}")
    em.emit(f"selp.f32 \t{out}, {over}, {f}, {big}", f"torch's {dt.name} overflow")
    return out

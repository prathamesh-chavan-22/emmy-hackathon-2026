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
        return self.mem == "f16"


DTYPES = {d.name: d for d in (
    DType("f32", 4, "f32", "f32", "f32"),
    DType("f16", 2, "f16", "f32", "noftz.f16"),
    DType("f64", 8, "f64", "f64", "f64"),
    DType("i32", 4, "s32", "s32", "s32"),
    DType("i64", 8, "s64", "s64", "u64"),   # two's complement: unsigned add is the same bits
)}

# C type names emmy uses for shared-memory arrays.
C_TYPES = {"float": "f32", "__half": "f16", "double": "f64", "int": "i32", "long long": "i64"}

INT_KINDS = ("s32", "s64")
FLOAT_KINDS = ("f16", "f32", "f64")


def get(name: str) -> DType:
    try:
        return DTYPES[name]
    except KeyError:
        raise NotImplementedError(f"dtype {name} (the runner supports {', '.join(DTYPES)})") from None


def bits(kind: str) -> int:
    return 16 if kind == "f16" else 64 if kind in ("s64", "f64") else 32


def common_kind(kinds) -> str:
    """The kind C's usual arithmetic conversions pick: double, else float, else the widest int."""
    kinds = set(kinds) - {"pred"}
    if "f64" in kinds:
        return "f64"
    if kinds & set(FLOAT_KINDS) or not kinds:
        return "f32"
    return "s64" if "s64" in kinds else "s32"


def cast_const(value, kind: str, rounds: bool = False):
    """A compile-time constant converted as the hardware cvt would.

    float -> int truncates and saturates (cvt.rzi; NaN -> 0), which is what
    emmy's `int acc = -1e30f` max identity becomes on the GPU.
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
        return float(np.float16(v)) if rounds else v


def cvt(em, dst: str, dst_kind: str, src: str, src_kind: str) -> None:
    """dst = src converted between numeric kinds, with C's rounding rules:
    float -> int truncates, int -> float and float narrowing round to nearest."""
    if src_kind == dst_kind:
        em.emit(f"mov.b{bits(dst_kind)} \t{dst}, {src}")
        return
    if dst_kind in INT_KINDS and src_kind in INT_KINDS:
        mode = ""
    elif dst_kind in INT_KINDS:
        mode = ".rzi"
    elif src_kind in INT_KINDS or bits(dst_kind) < bits(src_kind):
        mode = ".rn"
    else:
        mode = ""     # widening float: exact
    em.emit(f"cvt{mode}.{dst_kind}.{src_kind} \t{dst}, {src}")


def round_to(em, value: str, dt: DType) -> str:
    """An f32 register rounded to dt's precision (a new register), when dt rounds."""
    if not dt.rounds:
        return value
    h, f = em.reg("f16"), em.reg("f32")
    em.emit(f"cvt.rn.f16.f32 \t{h}, {value}")
    em.emit(f"cvt.f32.f16 \t{f}, {h}", f"round to {dt.name}")
    return f

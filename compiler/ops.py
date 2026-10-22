"""Elementwise op table: ElementwiseImpl name -> PTX sequence.

Each entry receives the emitter, the destination register and the argument
registers (f32, or .pred for bool values). Entries in CONST_OPS also get the
arguments' compile-time values (None when not constant), so e.g. pow(x, 3)
becomes two multiplies.

Accuracy matters more than speed here: pointwise tolerances can be 0 or 1e-6,
so transcendentals use range-reduced sequences (exp, erf), never a bare
ex2.approx on a large argument.
"""

from __future__ import annotations

import math

from .emitter import Emitter, f32_imm

LOG2E = f32_imm(math.log2(math.e))
LN2 = f32_imm(math.log(2.0))
ONE = f32_imm(1.0)
ZERO = f32_imm(0.0)

# erf polynomials, Horner order (highest power first); fitted by scripts/fit_erf.py,
# max error 1.3e-7 abs / 1.7e-7 rel in f32.
ERF_SMALL = (7.8824938100297e-05, -0.0008018853841349483, 0.005189311690628529, -0.02685432881116867,
             0.11283597350120544, -0.3761262595653534, 1.1283791065216064)
ERF_LARGE = (-4.778926410153872e-08, 1.1225415619264822e-06, -9.93936828308506e-06, 2.2871336113894358e-05,
             0.0003347133460920304, -0.004095306620001793, 0.025285322219133377, -0.10878679901361465,
             -0.6327647566795349, -1.1298409700393677, 0.00024826519074849784)
ERF_SATURATE = 3.92   # erfc(3.92) < 2^-25: erf is exactly 1 in f32 beyond this


def _simple(instr: str):
    def emit(em: Emitter, d: str, a: list[str]) -> None:
        em.emit(f"{instr} \t{d}, {', '.join(a)}")
    return emit


def _f(em: Emitter) -> str:
    return em.reg("f32")


def _horner(em: Emitter, coeffs, x: str) -> str:
    """c0*x^n + ... + cn as an fma chain."""
    acc = _f(em)
    em.emit(f"mov.f32 \t{acc}, {f32_imm(coeffs[0])}")
    for c in coeffs[1:]:
        nxt = _f(em)
        em.emit(f"fma.rn.f32 \t{nxt}, {acc}, {x}, {f32_imm(c)}")
        acc = nxt
    return acc


def _exp(em: Emitter, d: str, a: list[str]) -> None:
    """e^x to ~2 ulp, the sequence nvcc emits for expf.

    The naive ex2(x * log2e) loses up to 64 ulp for large |x|: the rounding error
    of the product is amplified by the exponent. Instead split x*log2e = n + f
    with integer n (clamped to [-126, 126]) and |f| <= 0.5, computing f with a
    two-part log2e so it keeps full precision; ex2.approx is accurate on small f,
    and 2^n is built directly as float exponent bits.
    """
    x = a[0]
    t, k, n, nn, f0, f, e, s = (_f(em) for _ in range(8))
    kb, sb = em.reg("b32"), em.reg("b32")
    em.emit(f"fma.rn.f32 \t{t}, {x}, 0f3BBB989D, 0f3F000000")   # x*log2e/252 + 0.5
    em.emit(f"cvt.sat.f32.f32 \t{t}, {t}")                     # clamp to [0, 1]
    em.emit(f"fma.rm.f32 \t{k}, {t}, 0f437C0000, 0f4B400001")   # 1.5*2^23 + 1 + floor(252 t)
    em.emit(f"add.f32 \t{n}, {k}, 0fCB40007F")                  # n = that - (1.5*2^23 + 127)
    em.emit(f"neg.f32 \t{nn}, {n}")
    em.emit(f"fma.rn.f32 \t{f0}, {x}, {LOG2E}, {nn}")            # f = x*log2e - n ...
    em.emit(f"fma.rn.f32 \t{f}, {x}, 0f32A57060, {f0}")         # ... + x*(log2e low part)
    em.emit(f"ex2.approx.f32 \t{e}, {f}")
    em.emit(f"mov.b32 \t{kb}, {k}")
    em.emit(f"shl.b32 \t{sb}, {kb}, 23")                        # low bits of k = n + 127 -> exponent
    em.emit(f"mov.b32 \t{s}, {sb}")
    em.emit(f"mul.f32 \t{d}, {e}, {s}", "e^x = 2^f * 2^n")


def _log(em: Emitter, d: str, a: list[str]) -> None:
    # ln x = log2(x) * ln 2; lg2.approx has ~2^-22 absolute error on the log2.
    t = _f(em)
    em.emit(f"lg2.approx.f32 \t{t}, {a[0]}")
    em.emit(f"mul.f32 \t{d}, {t}, {LN2}")


def _rsqrt(em: Emitter, d: str, a: list[str]) -> None:
    # Two correctly rounded steps (<= 1 ulp) instead of rsqrt.approx.
    s = _f(em)
    em.emit(f"sqrt.rn.f32 \t{s}, {a[0]}")
    em.emit(f"rcp.rn.f32 \t{d}, {s}")


def _sigmoid(em: Emitter, d: str, a: list[str]) -> None:
    # 1 / (1 + e^-x)
    n, e, s = _f(em), _f(em), _f(em)
    em.emit(f"neg.f32 \t{n}, {a[0]}")
    _exp(em, e, [n])
    em.emit(f"add.f32 \t{s}, {e}, {ONE}")
    em.emit(f"rcp.rn.f32 \t{d}, {s}")


def _silu(em: Emitter, d: str, a: list[str]) -> None:
    s = _f(em)
    _sigmoid(em, s, a)
    em.emit(f"mul.f32 \t{d}, {a[0]}, {s}")


def _tanh(em: Emitter, d: str, a: list[str]) -> None:
    # tanh x = 1 - 2 / (e^(2x) + 1): saturates cleanly to +-1 for large |x|.
    t, e, s, r = (_f(em) for _ in range(4))
    em.emit(f"add.f32 \t{t}, {a[0]}, {a[0]}")
    _exp(em, e, [t])
    em.emit(f"add.f32 \t{s}, {e}, {ONE}")
    em.emit(f"rcp.rn.f32 \t{r}, {s}")
    em.emit(f"fma.rn.f32 \t{d}, {r}, {f32_imm(-2.0)}, {ONE}")


def _erf(em: Emitter, d: str, a: list[str]) -> None:
    """Branch-free erf: evaluate both polynomial regions, select, restore the sign."""
    x = a[0]
    ax, t, small, c, e, large, r = (_f(em) for _ in range(7))
    em.emit(f"abs.f32 \t{ax}, {x}")
    em.emit(f"mul.f32 \t{t}, {ax}, {ax}")
    em.emit(f"mul.f32 \t{small}, {ax}, {_horner(em, ERF_SMALL, t)}", "|x| < 1: x * P(x^2)")
    em.emit(f"min.f32 \t{c}, {ax}, {f32_imm(ERF_SATURATE)}")
    _exp(em, e, [_horner(em, ERF_LARGE, c)])
    em.emit(f"sub.f32 \t{large}, {ONE}, {e}", "|x| >= 1: 1 - exp(Q(|x|))")
    p = em.reg("pred")
    em.emit(f"setp.lt.f32 \t{p}, {ax}, {ONE}")
    em.emit(f"selp.f32 \t{r}, {small}, {large}, {p}")
    em.emit(f"copysign.f32 \t{d}, {x}, {r}", "erf is odd")


def _pow(em: Emitter, d: str, a: list[str], consts: list) -> None:
    """x^y. A constant y that is a small integer or +-0.5 gets an exact sequence;
    anything else uses 2^(y log2|x|) with C pow's sign rules for negative x."""
    x, y = a
    k = consts[1]
    if k is not None and float(k).is_integer() and abs(k) <= 16:
        n = int(abs(k))
        if n == 0:
            em.emit(f"mov.f32 \t{d}, {ONE}")
            return
        # square-and-multiply: x^n with ~log2(n) roundings
        acc, base = None, x
        while True:
            if n & 1:
                if acc is None:
                    acc = base
                else:
                    nxt = _f(em)
                    em.emit(f"mul.f32 \t{nxt}, {acc}, {base}")
                    acc = nxt
            n >>= 1
            if not n:
                break
            sq = _f(em)
            em.emit(f"mul.f32 \t{sq}, {base}, {base}")
            base = sq
        em.emit(f"rcp.rn.f32 \t{d}, {acc}" if k < 0 else f"mov.f32 \t{d}, {acc}")
        return
    if k == 0.5:
        em.emit(f"sqrt.rn.f32 \t{d}, {x}")
        return
    if k == -0.5:
        _rsqrt(em, d, [x])
        return
    ax, l2, p, r, neg, yi, half, hi = (_f(em) for _ in range(8))
    em.emit(f"abs.f32 \t{ax}, {x}")
    em.emit(f"lg2.approx.f32 \t{l2}, {ax}")
    em.emit(f"mul.f32 \t{p}, {y}, {l2}")
    em.emit(f"ex2.approx.f32 \t{r}, {p}", "|x|^y")
    # negative x: integer y keeps the magnitude (negated when odd), otherwise NaN
    is_int, is_odd, x_neg, flip, y_zero = (em.reg("pred") for _ in range(5))
    em.emit(f"cvt.rni.f32.f32 \t{yi}, {y}")
    em.emit(f"setp.eq.f32 \t{is_int}, {yi}, {y}")
    em.emit(f"mul.f32 \t{half}, {y}, 0f3F000000")
    em.emit(f"cvt.rni.f32.f32 \t{hi}, {half}")
    em.emit(f"setp.ne.and.f32 \t{is_odd}, {hi}, {half}, {is_int}")
    em.emit(f"setp.lt.f32 \t{x_neg}, {x}, {ZERO}")
    em.emit(f"and.pred \t{flip}, {x_neg}, {is_odd}")
    em.emit(f"neg.f32 \t{neg}, {r}")
    em.emit(f"selp.f32 \t{r}, {neg}, {r}, {flip}")
    em.emit(f"@{x_neg} selp.f32 \t{r}, {r}, 0f7FFFFFFF, {is_int}", "negative ^ non-integer = NaN")
    em.emit(f"setp.eq.f32 \t{y_zero}, {y}, {ZERO}")
    em.emit(f"selp.f32 \t{d}, {ONE}, {r}, {y_zero}", "x^0 = 1")


def _cmp(cond: str):
    def emit(em: Emitter, d: str, a: list[str]) -> None:
        em.emit(f"setp.{cond}.f32 \t{d}, {a[0]}, {a[1]}")
    return emit


def _where(em: Emitter, d: str, a: list[str]) -> None:
    em.emit(f"selp.f32 \t{d}, {a[1]}, {a[2]}, {a[0]}")


def _sign(em: Emitter, d: str, a: list[str]) -> None:
    # copysign(1, x) for x != 0, else 0 (NaN stays NaN through the compare)
    s, p = _f(em), em.reg("pred")
    em.emit(f"copysign.f32 \t{s}, {a[0]}, {ONE}")
    em.emit(f"setp.eq.f32 \t{p}, {a[0]}, {ZERO}")
    em.emit(f"selp.f32 \t{d}, {a[0]}, {s}, {p}")


F32_OPS = {
    "add": _simple("add.f32"),
    "subtract": _simple("sub.f32"),
    "multiply": _simple("mul.f32"),
    "divide": _simple("div.rn.f32"),
    "true_divide": _simple("div.rn.f32"),
    "maximum": _simple("max.f32"),
    "minimum": _simple("min.f32"),
    "negative": _simple("neg.f32"),
    "abs": _simple("abs.f32"),
    "absolute": _simple("abs.f32"),
    "sqrt": _simple("sqrt.rn.f32"),
    "rsqrt": _rsqrt,
    "reciprocal": _simple("rcp.rn.f32"),
    "copy": _simple("mov.f32"),
    "square": lambda em, d, a: em.emit(f"mul.f32 \t{d}, {a[0]}, {a[0]}"),
    "relu": lambda em, d, a: em.emit(f"max.f32 \t{d}, {a[0]}, {ZERO}"),
    "floor": _simple("cvt.rmi.f32.f32"),
    "ceil": _simple("cvt.rpi.f32.f32"),
    "trunc": _simple("cvt.rzi.f32.f32"),
    "rint": _simple("cvt.rni.f32.f32"),
    "round": _simple("cvt.rni.f32.f32"),      # torch.round is half-to-even, like rni
    "sign": _sign,
    "exp": _exp,
    "exp_fast": _exp,
    "exp2": _simple("ex2.approx.f32"),
    "log": _log,
    "log2": _simple("lg2.approx.f32"),
    "sigmoid": _sigmoid,
    "silu": _silu,
    "tanh": _tanh,
    "erf": _erf,
    "where": _where,
    # comparisons produce .pred (IR dtype bool)
    "greater": _cmp("gt"),
    "greater_equal": _cmp("ge"),
    "less": _cmp("lt"),
    "less_equal": _cmp("le"),
    "equal": _cmp("eq"),
    "not_equal": _cmp("neu"),     # unordered: NaN != NaN is true, as in C
}

# Boolean ops on .pred values. torch spells `mask_a & mask_b` as bitwise_and on bools.
PRED_OPS = {
    "logical_and": _simple("and.pred"),
    "logical_or": _simple("or.pred"),
    "logical_xor": _simple("xor.pred"),
    "logical_not": _simple("not.pred"),
    "bitwise_and": _simple("and.pred"),
    "bitwise_or": _simple("or.pred"),
    "bitwise_xor": _simple("xor.pred"),
    "bitwise_not": _simple("not.pred"),
    "invert": _simple("not.pred"),
}

CONST_OPS = {
    "power": _pow,
    "pow": _pow,
}


def emit_op(em: Emitter, name: str, dst: str, args: list[str], consts: list) -> None:
    if name in CONST_OPS:
        CONST_OPS[name](em, dst, args, consts)
    elif name in F32_OPS:
        F32_OPS[name](em, dst, args)
    elif name in PRED_OPS:
        PRED_OPS[name](em, dst, args)
    else:
        raise NotImplementedError(f"elementwise op {name!r}")

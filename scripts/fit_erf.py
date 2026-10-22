#!/usr/bin/env python3
"""Fit the erf polynomials used by compiler/ops.py and report their f32 error.

    python scripts/fit_erf.py

Two regions, the standard split for single-precision erf:
  |x| < 1          erf(x) = x * P(x^2)          (P: degree 6 in x^2)
  1 <= |x| < 3.92  erf(x) = 1 - exp(Q(|x|))     (Q: degree 10, fits log(erfc), which is smooth)
  |x| >= 3.92      erfc < 2^-25, so erf rounds to exactly 1 in f32 (Q is evaluated at 3.92)
Least squares on Chebyshev nodes in float64; the check evaluates Horner in float32.
"""

import math

import numpy as np

A_DEG, B_DEG, B_HI = 6, 10, 3.92


def cheb_nodes(lo, hi, n=2000):
    k = np.arange(n)
    return (lo + hi) / 2 + (hi - lo) / 2 * np.cos(np.pi * (k + 0.5) / n)


def fit():
    t = cheb_nodes(1e-6, 1.0)                     # t = x^2
    x = np.sqrt(t)
    a = np.polyfit(t, [math.erf(v) / v for v in x], A_DEG)
    xb = cheb_nodes(1.0, B_HI)
    b = np.polyfit(xb, [math.log(math.erfc(v)) for v in xb], B_DEG)
    return a, b                                    # highest power first (Horner order)


def horner32(coeffs, v):
    acc = np.float32(coeffs[0])
    for c in coeffs[1:]:
        acc = np.float32(acc * v + np.float32(c))
    return acc


def erf32(x, a, b):
    ax = np.float32(abs(x))
    if ax < 1:
        r = np.float32(ax * horner32(a, np.float32(ax * ax)))
    else:
        q = horner32(b, np.float32(min(ax, np.float32(B_HI))))
        r = np.float32(1.0 - np.float32(math.exp(float(q))))
    return math.copysign(float(r), x)


if __name__ == "__main__":
    a, b = fit()
    xs = np.concatenate([np.linspace(-6, 6, 200001), np.geomspace(1e-8, 1, 2000)]).astype(np.float32)
    err_abs = max(abs(erf32(float(v), a, b) - math.erf(float(v))) for v in xs)
    err_rel = max(abs(erf32(float(v), a, b) - math.erf(float(v))) / abs(math.erf(float(v)))
                  for v in xs if v != 0)
    print(f"max abs err {err_abs:.3e}, max rel err {err_rel:.3e} (f32 ulp = 6.0e-08)")
    fmt = lambda cs: ", ".join(f"{float(np.float32(c))!r}" for c in cs)
    print(f"ERF_SMALL = ({fmt(a)})")
    print(f"ERF_LARGE = ({fmt(b)})")

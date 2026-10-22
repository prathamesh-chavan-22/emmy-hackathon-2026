# Compiler log (video script source)

## Results table

| date | change | p00 | p01 | p02 | p03 | r01 | r02 | r03 | r04 | m01 | m02 | f01 | f02 | f03 | a01 | a02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-22 | skeleton + phase 1 nodes | PASS | PASS | PASS | PASS | - | - | - | - | - | - | - | - | - | - | - |
| 2026-10-22 | robustness pass (accurate ops, typed Assign, consts) | PASS | PASS | PASS | PASS | - | - | - | - | - | - | - | - | - | - | - |

Extra test set (tests/problems, generated from tests/expressions_extra.py): pointwise 35/35 PASS;
reductions/fused 10 waiting on phase 2.

## 2026-10-22 - skeleton, category 1

- Layout: `compile.py` (CLI) -> `compiler/loader.py` (Graph.from_dict + meta.json) ->
  `prologue.py` (header, signature in arg_order, ld.param + cvta.to.global) ->
  `stmts.py` (dispatch table by statement class) -> `expr.py` (index/predicate trees) ->
  `emitter.py` (fresh virtual regs, `.reg` decls sized at the end).
  `context.py` holds per-kernel state (SSA name -> register, buffers); `ops.py` is the
  elementwise op table.
- Nodes: Tile, Load (v4/v2 vectorized), Assign, Let, Write (vector st, red.global.add for atomic).
- Tile: flat cell = ctaid.x*ntid.x+tid.x, guard `cell < N` (N = product of axis extents,
  not in the IR), row-major decode of axes. If the launch has fewer threads than cells,
  the Tile becomes a grid-stride loop.
- Expressions fold at compile time when constant (all shapes are static), so `a0*2+0`
  becomes one multiply. Multi-dim indices flatten row-major against the meta.json shape.
- Constant byte offsets go into `[reg+imm]` addressing.
- Bug: emmy shapes are `Dim` objects, not ints (`Dim.value`).
- Flipped to PASS: p00, p01, p02 (ConstantOp args), p03 (relu).

## 2026-10-22 - robustness pass + our own test set

- **Test generation.** The hidden set is made by tools/generate.py from a private expression
  list, so we do the same with our own list (tests/expressions_extra.py, 45 problems) in a
  separate torch venv - test-only, never in compile.py. Windows needed PYTHONUTF8=1, an absolute
  --emmy path and a no-op fcntl stub.
- **What emmy cannot trace tells us what the hidden set cannot contain:** ceil, sign, leaky_relu,
  elu, hardtanh, min(), `3.0 - x` (rsub) all fail inside emmy 0.3.11.
- **tests/check_local.py**: offline gate (no GPU) - ASCII, header, entry name, param count,
  compile time, for every IR.
- **Accuracy.** p00/p03 tolerate 0 error, so pointwise must be exact where f32 allows it:
  - exp: naive `ex2.approx(x*log2e)` is up to 64 ulp off at large |x| (measured). Replaced by
    nvcc's expf sequence: split x*log2e into integer n + small f with a two-part log2e,
    ex2 on f only, 2^n from exponent bits. ~2 ulp. Passes exp(20*x) with rtol 1e-6.
  - rsqrt: sqrt.rn + rcp.rn instead of rsqrt.approx.
  - erf (for gelu): own polynomials, fitted by scripts/fit_erf.py: x*P(x^2) below 1,
    1-exp(Q(|x|)) above, 1 beyond 3.92. Max error 1.7e-7 relative.
- **Typed values.** bool IR values live in .pred registers: comparisons -> setp, where -> selp,
  `mask & mask` (bitwise_and on bools) -> and.pred.
- **ConstantOp folding.** Constant values are in the IR, so a load of a constant buffer becomes
  an immediate (no memory access) and pow(x, 3.0) becomes square-and-multiply; fractional pow
  falls back to 2^(y log2|x|) with C sign rules.
- **Launch with fewer threads than Tile cells** (grid-stride Tile) never occurs in emmy output,
  so a synthetic test (x_p_short_launch: p01 on 1 block) now covers that path.
- Also: scalar by-value params (`"scalar": true` in meta), clear errors for unknown dtypes and
  ops, UTF-8 reads in the loader (Windows default codepage would break on non-ASCII IR).

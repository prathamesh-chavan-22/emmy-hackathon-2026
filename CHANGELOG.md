# Compiler log (video script source)

## Results table

| date | change | p00 | p01 | p02 | p03 | r01 | r02 | r03 | r04 | m01 | m02 | f01 | f02 | f03 | a01 | a02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-22 | skeleton + phase 1 nodes | PASS | PASS | PASS | PASS | - | - | - | - | - | - | - | - | - | - | - |
| 2026-10-22 | robustness pass (accurate ops, typed Assign, consts) | PASS | PASS | PASS | PASS | - | - | - | - | - | - | - | - | - | - | - |
| 2026-10-22 | phase 2: loops, Accum, WarpShuffle, Smem, Sync, Cond, TreeHalve | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | - | - | PASS | PASS | PASS | PASS | PASS |
| 2026-10-22 | phase 3: cp.async, dynamic smem pool | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

Extra test set (tests/problems, generated from tests/expressions_extra.py): 50/50 PASS
(pointwise incl. sin/cos + leaky/hardtanh/elu compositions, reductions, unpinned atomic sum).

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

## 2026-10-22 - phase 2: cooperative reductions (13/15 open problems)

- **Loops.** StridedLoop and Loop share one counted-loop lowering: loop register = start,
  top-tested `setp.ge` against the end, body, `add step`, branch back. Accumulators folded
  directly in the loop body are seeded with their op's identity (0 for add, -inf for max)
  just before the loop.
- **Mutable vs SSA names.** Most IR names are SSA: assigning just rebinds the name to a new
  register. Loop-carried values (accumulators, Init state) must stay in ONE register across
  iterations, so the context marks them mutable and assignments become `mov` into that register.
- **WarpShuffle** = `shfl.sync.bfly.b32` with offsets 16,8,4,2,1 and `activemask` as the member
  mask (a small block can be a partial warp). After the butterfly every lane has the warp total.
- **Smem** = a `.shared` array; its address is a 32-bit offset, so shared Load/Write use 32-bit
  address math and `ld/st.shared`. Same Load/Write code path as global, different state space.
- **TreeHalve** = per-warp partials in shared memory halved level by level (unrolled, since the
  length is static): threads t < s fold buf[t+s] into buf[t], `bar.sync` between levels, then
  every thread reads buf[0].
- **Cond** = `@!p bra else`; statically-known conditions emit only the live side.
- `lane` = tid & 31, `warp` = tid >> 5, bound when the Tile starts.
- Fused (rmsnorm, softmax, l2norm) and attention needed nothing new: they are the same nodes
  composed. Only matmul (cp.async) is left.
- Test harness: the shared runner times out under load -> run_all retries connect timeouts.

## 2026-10-22 - phase 3: matmul (15/15 open problems)

- **cp.async**: CpAsyncCopy -> `cp.async.cg.shared.global [smem], [gmem], 16` (16-byte copies
  bypass L1; 4/8-byte use `.ca`). Commit -> `cp.async.commit_group`, Wait(n) ->
  `cp.async.wait_group n`: m01 keeps 2 tiles in flight (a 3-stage pipeline) while computing a third.
- **Dynamic shared memory**: m01 needs 72 KB of shared memory, over the 48 KB static limit. When
  meta.json gives smem_bytes > 0, all Smem arrays are carved from one
  `.extern .shared .align 16 .b8 dyn_smem[]` pool at aligned offsets (checked against smem_bytes).
- **Bug found**: a loop with seed=True was releasing accumulators that an enclosing Init owned;
  now a loop only releases the accumulators it seeded itself.
- m02 (split-K) needed nothing else: atomic writes + zero_outputs already worked.

## 2026-10-22 - more diverse tests, Select, review round 3 (15/15 open, 73/73 extra)

- **23 new test problems** aimed at the thin categories: matmul (32x32, tall, wide, K=2048
  split-K, odd 100x70x50, 256x384x256, batched 4x, bias+relu epilogue, scaled), fused (full
  layernorm, log-softmax, scaled/odd softmax, rmsnorm+silu, max-subtract), attention (32x16,
  256x64, rectangular Q/K/V, batched 4 heads), reductions (3-D mean, dim-0 sum, abs-max,
  unpinned rows). All pass.
- **New node: Select** (the odd matmul masks its out-of-range tail with it) = a chain of
  `selp`, first true predicate wins, last branch is the catch-all.
- **Test-writing gotcha**: emmy's tracer turns a named intermediate (`d = x - m`) into a new
  random input, so fused tests must write intermediates inline. log_softmax and prod do not
  trace at all -> the hidden set cannot contain them.
- **Review fixes**: if/else bodies and warp-shuffle steps are now scopes (names bound inside
  no longer leak out); address math is one `mad` instead of mul + add.

## 2026-10-23 - op/statement audit, bool<->float masks (15/15 open, 85/85 extra)

The question: what can the HIDDEN set contain that we have never seen? It is generated by
the same emmy, so emmy itself is the boundary. Three checks:

- **Ops**: a hidden kernel can only use ops emmy's own CUDA renderer accepts (`op_to_expr`).
  We cover all of them; added `pad` (an identity, like `copy`). Probed ~45 torch ops: floor,
  ceil, round, tan, atan, log10, hypot, lerp, var, std, norm, logsumexp, argmax, amin,
  logical_not, ... are all rejected by emmy, so they cannot appear. Everything emmy accepts
  compiles with our compiler.
- **Tensor cores** (mma / ldmatrix / wgmma statements): emmy only picks them for f16/bf16/fp8
  inputs; for f32 there is no atom on any GPU. Not reachable.
- **ZeroPrologue**: added in emmy's cuda-stage lowering, after the kernel-IR dump we are
  given, and not even loadable by `Graph.from_dict`. Not reachable. (Wrote a handler, proved it
  dead, removed it.)

**Bug found by the audit**: emmy's C casts bool <-> float implicitly (`x * (x > 0)`,
`(x > 0.5).float()`); we emitted `mul.f32 %f, %f, %p` = invalid PTX. Fix: each op declares the
kinds it computes in (`ops.signature`), and `expr.convert` applies C's rules at the boundary
(nonzero -> true, true -> 1.0). PTX for every earlier test is byte-identical.

New tests: 4 mask mixes, cumsum (a scan) over 1-D and rows, softmax over dim 0, mean over the
middle axis, softsign, tanhshrink, pow with a tensor exponent, negative integer power.

## 2026-10-23 - speed: unrolling and constant address offsets (15/15 open, 85/85 extra)

Same schedules emmy chose; better instructions inside them. Measured on the runner (RTX 5090):

| problem | before | after |
|---|---|---|
| m01 matmul 512 | 114 us | ~78 us |
| a01 attention | 218 us | ~171 us |
| a02 attention 128 | 100 us | ~54 us |
| r02 big sum | 39 us | ~31 us |

- **Affine index lowering** (memory.py): an index is split into (register part, constant
  part); the constant rides PTX's free `[reg+imm]` address offset instead of costing an add.
- **Full unroll** of loops emmy marks `unroll` (its `#pragma unroll`, e.g. a matmul's k-step):
  the loop variable becomes a constant, so every address in the body shares one base register.
- **Partial unroll** (8 copies per trip, plus a remainder loop) of other innermost loops,
  including strided ones: 8 loads in flight instead of one latency-bound chain. Each thread
  still does the same operations in the same order, so results are bit-identical.
- Lesson: unrolling a loop that CONTAINS loops ran attention out of registers (255, spilling,
  5x slower), so only innermost loops are unrolled. And ptxas is sensitive: a needless
  `i + 7 < 32` trip guard (instead of `i < 32`) cost a02 25%.

## 2026-10-23 - shape sweep, exact rounding, review round 4 (15/15 open, 86/86 extra, 43/43 sweep)

- **Shape sweep** (tests/expressions_sweep.py): 43 hidden-set-like problems at sizes the open
  set never shows: 1000^3 / 1024^3 / 1x512x512 / 512x512x1 / 33x65x17 matmuls, attention up to
  1024 tokens and head dim 128, sums over 1M and a prime length, reductions with emmy's default
  schedule as well as the pinned one, big softmax / layernorm / log-softmax. All pass.
- **Bug found by the sweep**: `a * b + c` was off in the last bit. Our `mul.f32` / `add.f32`
  carry no rounding mode, so ptxas may fuse them into one fma (one rounding instead of two).
  numpy rounds twice, and a pointwise problem can be checked exactly. Fix: the IR's own
  arithmetic uses `.rn` (never fused, bit-exact with numpy); accumulations `acc += a*b` emit
  `fma.rn` explicitly, as nvcc does for emmy's C, so matmul keeps its speed.
- **Review round 4 fixes**: full unroll capped at 64 copies; strided loops with too few trips stay
  rolled; rebinding a name drops its `i + k` addressing form; bool -> float writes use the
  register kind; literal TABs in emitted strings back to `\t`.
- Not changed: comparisons on index ints go through f32 (emmy's C promotes them the same way);
  `any`/`all` folds (emmy cannot trace them).

## 2026-10-23 - other dtypes: f16, f64, i32, i64 (15/15 open, 119/119 extra, 43/43 sweep)

- **What the hidden set can hold**: the runner allocates only f32, f16, f64, i32 and i64
  (docs/API.md), so bf16 / int8 / fp8 / fp4 problems cannot exist. tools/generate.py now
  rejects them with that reason instead of a KeyError.
- **One table, compiler/dtypes.py**: per dtype its size, its memory type, the register kind it
  computes in and its atomic type. Loads and stores convert between memory and compute kind, so
  the statement lowerers did not need per-dtype code.
- **How emmy's C types things** (read from its renderer, then checked on its reference kernels):
  every variable keeps its declared type, but nearly all math is float. We mirror that, except
  where emmy and torch disagree. The expected outputs come from torch, so there we side with torch:
  - f16 values sit in f32 registers and each f16-typed result is rounded to half on assignment.
    For + - * / sqrt that is bit-identical to native half math (f32 has more than twice the bits).
  - f64: emmy narrows to float for all math. We compute + - * / sqrt min max natively in f64,
    so those match torch exactly; transcendentals go through f32 as in emmy.
  - i32/i64: native integer add/sub/mul/shift/bitwise ops, and other ops through f32 with C's
    truncation back to int. emmy writes `a // b` and `a % b` as C `/` and `%`, which truncate,
    while torch floors. With signed inputs they disagree on about half the elements, so we floor.
  - An int accumulator seeded with emmy's `-1e30f` max identity saturates to INT_MIN, as the
    GPU's cvt does; shuffles and smem trees carry it as float, as emmy's C does.
- **33 new tests** (x_d_*): f16 pointwise/softmax/sum/mean/conversions, f64 pointwise/softmax/sum/matmul,
  i32/i64 arithmetic, floor div/mod on negatives, bitwise, compares, max/sum/cumsum. All pass.
- Found while testing: submit.py reads every output back as float32, so an f64 result is
  checked to f32 precision at best.
- Not done yet: f16 matmul and attention, which emmy lowers to tensor cores (mma.sync, ldmatrix,
  swizzled shared memory).

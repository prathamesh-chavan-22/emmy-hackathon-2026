# Emmy Kernel IR -> PTX compiler (submission)

`python compile.py path/to/kernel.ir.json > out.ptx`

This is a compiler from Emmy Kernel IR to PTX for sm_89, written from scratch in about 2,400 lines of
Python (`compiler/`). emmy is used only to load the IR (`Graph.from_dict`). Every PTX instruction
comes from our own code: no nvcc, no emmy renderer, and no other compiler in the path. The step-by-step
development log, with the bugs we found and the measurements, is [CHANGELOG.md](CHANGELOG.md).

## Results

| set | what | pass |
|---|---|---|
| open problems | the 15 in `problems/` (all 5 categories) | 15/15 |
| extra tests | 172 of our own, generated like the hidden set (`tests/expressions_extra.py`, `_llm.py`, `_quant.py`) | 172/172 |
| shape sweep | 43 hidden-set-like sizes (`tests/expressions_sweep.py`) | 43/43 |

These were all run on the hosted RTX 5090 runner through `submit.py --compiler`.

## Features

**All five categories.**
- **Pointwise:** every elementwise op emmy's renderer accepts (about 50 primitives, plus pow, bitwise and integer ops), broadcasting, multi-output kernels, grid-stride Tiles.
- **Reductions:** warp shuffles, shared-memory tree reductions, split-K atomics (`red.global.add`),
  scans (cumsum).
- **Matmul:** `cp.async` multi-stage pipelines, dynamic shared-memory pool above 48 KB.
- **Fused:** layernorm, rmsnorm, softmax, log-softmax.
- **Attention:** online-softmax attention, sdpa, causal and masked.
- **Modern-LLM blocks** (tests): Mamba-2 SSM decay and scan, causal linear attention, SwiGLU,
  RoPE, weighted RMSNorm, logit softcapping, grouped-query attention.

**Data types:** f32, f16, f64, i32 and i64, with torch's semantics where emmy's C and torch
disagree (integer division, f16 rounding of intermediates, bool <-> float masks).

**Low precision** (inside kernels; the runner allocates only f32/f16/f64/i32/i64 buffers):
- bf16 values, rounded on every assignment (`cvt.rn.bf16.f32`), bit-exact against torch.
- fp8 e4m3 and e5m2, bit-exact with torch's rounding, subnormals and overflow rules (e4m3fn
  saturates, e5m2 goes to inf). The hardware cvt needs PTX 8.1, so we round in plain f32 with a
  magic-constant add that makes `add.rn` round at the fp8 step.
- int4 dequantization from packed i32 words.
- int8 and fp4 are not reachable: emmy 0.3.11 has no int8 dtype, and its fp4 has no torch cast.

**Tensor cores (f16 and bf16):**
- `mma.sync.m16n8k16` with an f32 accumulator, f16 or bf16 operands
  (`a.to(torch.bfloat16) @ b.to(torch.bfloat16)` runs on bf16 tensor cores).
- Five `ldmatrix` variants (x4, x2, transposed, paired), plus fragments read straight from global
  memory, with edge clamping and zero-filled K.
- Swizzled shared memory (B32/B64/B128) for both the cp.async fill and the ldmatrix drain.
- Fused epilogues (bias, relu, gelu, residual) on the accumulator registers.
- Attention's softmax computed entirely in registers: fragment apply, row max/sum over the 4 lanes
  of a row, and repacking scores into the next mma's operand.

**Accuracy:**
- Correct rounding (`.rn`) throughout, and `fma` where nvcc contracts emmy's C.
- Our own `exp` (about 2 ulp, nvcc's range reduction).
- Fitted `erf` polynomials (1.7e-7 relative; `scripts/fit_erf.py`).
- Exact square-and-multiply for integer `pow`.

**Speed:** we keep the schedule emmy chose and generate better instructions inside it.
- Affine index splitting: constant offsets ride the free `[reg+imm]` address field.
- Vectorized `ld/st.v2/.v4`.
- Full unrolling of loops emmy marks `#pragma unroll`, and 8-way partial unrolling of other
  innermost loops.
- Constants folded from ConstantOp buffers.
- `.maxntid` launch bounds.

Measured: the 512 matmul went from 114 to 78 us, and the 128-token attention from 100 to 54 us.

**Robustness:**
- Always a bounds guard.
- Pure-ASCII output, checked at emit time.
- Clear errors for constructs emmy can emit but we do not lower (fp8 in memory, fp8 mma).

## Layout

| file | role |
|---|---|
| `compile.py` | command line entry |
| `compiler/loader.py` | `Graph.from_dict` and meta.json (param order, launch) |
| `compiler/prologue.py` | PTX header, signature in `arg_order`, `ld.param`, `.maxntid` |
| `compiler/stmts.py` | the statement dispatch table: Tile, Loop, Load/Write, Accum, Cond, ... |
| `compiler/expr.py` | index and value expressions, constant folding, C conversion rules |
| `compiler/memory.py` | addressing, vector loads/stores, swizzle |
| `compiler/coop.py` | cross-thread code: shared memory, cp.async, barriers, warp shuffles, tree reductions |
| `compiler/mma.py` | tensor cores: fragments, mma.sync, ldmatrix, fragment softmax ops |
| `compiler/ops.py`, `dtypes.py` | elementwise op table and per-dtype semantics |
| `compiler/emitter.py` | virtual registers and declarations (ptxas allocates registers) |

Every lowering function has a docstring explaining *why* it lowers the way it does.

## Tests and tools

- `tests/check_local.py`: an offline gate on every test IR, no GPU needed. It checks ASCII,
  header, entry name, parameters and compile time.
- `tests/run_all.py [--extra] [--sweep] [filter]`: runs the tests on the hosted runner.
- `tools/generate.py`: builds test problems from torch expressions through emmy, the same way the
  hidden set is made. It now writes multi-kernel problems in dependency order.

## Findings about emmy (from testing)

- With a head dim that is not a multiple of 16, emmy's attention tiling skips part of K.
  Its own reference kernel has the same error.
- Many torch ops (ceil, sign, leaky_relu, log_softmax, ...) do not trace through emmy 0.3.11, so the
  hidden set cannot contain them.
- torch 2.14 saturates e4m3fn overflow to +-448 (even inf); the runner returns inf as JSON null,
  so a kernel producing inf cannot pass there.
- An f16 `(a*b).sum()` and an f16 vector matmul produce identical IR but differ in torch's
  rounding. Only the kernel name tells them apart.

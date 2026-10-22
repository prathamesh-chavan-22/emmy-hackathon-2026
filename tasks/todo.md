# Status briefing (2026-10-22, deadline Oct 27 23:59 AoE)

## Score: 15/15 open problems pass. 50/50 extra tests pass.

| cat | problems | status |
|---|---|---|
| 1 pointwise | p00 p01 p02 p03 | PASS (exact match) |
| 2 reductions | r01 r02 r03 r04 | PASS |
| 3 matmul | m01 m02 | PASS |
| 4 fused | f01 f02 f03 | PASS (no new nodes needed) |
| 5 attention | a01 a02 | PASS (no new nodes needed) |

Run tests: `py tests/run_all.py` (or `py tests/run_all.py r0` for one category).
Python: `py` = 3.14. Repo: github.com/prathamesh-chavan-22/emmy-hackathon-2026

## DONE

- [x] Repo cloned to F:\cloudrift emmy competition, deps installed, sanity check passes
- [x] Read README, TASK, MVP, docs/API, PROVENANCE, tools/expressions_open.py, example PTX
- [x] Compiler skeleton (all files below) + tests/run_all.py + CHANGELOG.md (video notes)
- [x] Phase 1: Tile, Load, Assign, Let, Write -> p00-p03 pass

## HOW IT WORKS (know this for the video)

`py compile.py X.ir.json > out.ptx` runs this pipeline:

1. **loader.py** - `Graph.from_dict` (only emmy use), find the KernelOp, read meta.json
   next to the IR: param order (`arg_order`), shapes, grid/block.
2. **prologue.py** - writes `.version 8.0 / .target sm_89 / .address_size 64`, the
   `.entry <name>(...)` signature (one `.u64` pointer per buffer, constants included),
   `ld.param` + `cvta.to.global` for each pointer, global thread id = ctaid*ntid + tid.
3. **stmts.py** - walks the IR statement tree; dispatch table class name -> function:
   - Tile: N = product of axis sizes; guard `if cell >= N skip` (NOT in the IR, we add it);
     split the flat cell id into axis indices (row-major); if fewer threads than cells,
     loop (grid-stride).
   - Load/Write: compute address = base + flat_index*4; 2 or 4 consecutive values ->
     one vector instruction `ld/st.global.v2/v4.f32`; atomic write -> `red.global.add.f32`.
   - Assign: elementwise op via ops.py. Let: bind an expression to a name.
4. **expr.py** - index math (`a0*2+1`) lowered recursively. Constants folded at compile
   time (shapes are static), constant offsets go into `[reg+8]` addressing.
5. **emitter.py** - fresh virtual registers %f1 %f2 %r1 %rd1 %p1..., declared at the top.
   No register allocation - ptxas does it.
6. **ops.py** - table: op name (add, multiply, relu, exp, sigmoid...) -> PTX instructions.
   **context.py** - map of IR name -> register, plus the buffers.

## DONE: robustness + diverse tests (2026-10-22) - 39/39 pointwise pass, see CHANGELOG

Idea: the hidden set is made by tools/generate.py from a private expression list. We do the
same with our OWN expression list -> real emmy IRs + meta.json + expected.npz. Test-only:
torch/`emmy compile` never touch compile.py (rule-safe).

1. [x] Generator env: separate venv `.venv-gen` (torch CPU + emmy-ml 0.3.11), gitignored.
       Keeps the submission env torch-free, so compile.py can't accidentally depend on it.
       Feasibility check first: trace ONE expression. (Risk: if `emmy compile --ir cuda`
       needs nvcc, fall back to IR-only dumps.)
2. [x] `tests/expressions_extra.py` (same format as tools/expressions_open.py) -> `tests/problems/`
       Pointwise (should pass now - this is the robustness test):
       - odd sizes (7, 33, 1000, 1001): unroll tails, scalar + vector mixes
       - big sizes (1M): more cells than threads -> grid-stride loop path
       - 2-D / 3-D shapes, broadcasting ((64,128)+(128,), (64,1)*(64,128))
       - ops: sub, div, neg, abs, exp, log, sqrt, rsqrt, sigmoid, tanh, silu, gelu,
         max/min, clamp, where, square, pow, reciprocal; long chains; 3+ inputs; many constants
       - two outputs from one expression (multi-kernel / multi-write)
       - tolerances tight (0 for exact ops, 1e-6 for transcendentals) = worst case
       Reductions (expected to fail until phase 2, used as its test set):
       - sum/mean/max/min, odd lengths, row reductions of varying width, pinned + unpinned
3. [x] `tests/check_local.py`: offline, no GPU - compile every IR (open + extra), check ASCII,
       header, entry name, param count == arg_order, compile time < 1 s. Fast gate.
4. [x] run_all.py: `--extra` flag to run tests/problems too.
5. [x] Fix whatever breaks, plus the known gaps: accurate exp/log/tanh/sigmoid/rsqrt,
       scalar by-value args, clear error on non-f32 dtype, missing op names (gelu/erf, where...).
6. [x] Log results + bugs in CHANGELOG.md.

## TO DO (in order)

### Quick fixes (DONE in robustness pass)
- [x] ops.py: exp / sigmoid / tanh / rsqrt use fast approximations. Pointwise tolerance can
      be 0 (p00/p03 are exact) -> switch to accurate versions or hidden pointwise may fail.
- [x] Scalar (by-value) args in signature (`"scalar": true` in meta) - cheap insurance.
- [x] Unknown dtype -> clear error (only f32 supported; all problems are f32).

### Phase 2 - reductions r01-r04 (= MVP)
Test set: r01-r04 + tests/problems/x_r_* (sum100, sum1000, max, sumsq, mean_rows, rowsum_wide,
rowmax, colsum, unpinned_sum) -> `py tests/run_all.py --extra r0 x_r`
- [x] StridedLoop (loop: start, step, end), Accum (acc += x, starts at op identity)
- [x] WarpShuffle: `shfl.sync.bfly.b32` ladder 16,8,4,2,1
- [x] Smem (`.shared` array), Sync (`bar.sync 0`), Cond (`setp` + branch)
- [x] TreeHalve (combine per-warp partials through shared memory)
- [x] Built-in vars `lane` = tid%32, `warp` = tid/32
- [x] Watch out: guard branch + bar.sync = hang if some threads skip the barrier
- [ ] THEN: commit, push to YOUR public GitHub repo, submit once at
      riftstack.ai/hackathon/submit (MVP secured)

### Phase 3 - matmul m01 m02
- [x] Loop, Init, 2-D indexing, accumulator tiles
- [x] CpAsyncCopy / Commit / Wait (`cp.async` global -> shared)
- [x] Atomics + zero_outputs (m02); dynamic shared memory when meta smem_bytes > 0

### Phase 4 - fused f01 f02 f03
- [ ] Reduction + elementwise in one kernel (rmsnorm, l2norm)
- [ ] Online softmax: two linked accumulators (max + sum)

### Phase 5 - attention a01 a02
- [ ] Softmax fused into matmul

### Before submitting (deadline Oct 27)
- [ ] Public GitHub repo with compile.py at root
- [ ] 3-5 min video in YOUR voice explaining the compiler (CHANGELOG.md = script).
      Rule: if it looks like the AI understood it and you didn't -> no prize.
- [ ] Final run of all 15, update CHANGELOG results table

## RULES (disqualification if broken)
- Lowering must be our own code. emmy only for Graph.from_dict.
  No render_kernelop, no `emmy compile`, no nvcc.
- PTX pure ASCII. Entry name = IR kernel name. Param order = meta.json arg_order.

## REVIEW (2026-10-22, after phase 3)

Result: 15/15 open problems + 50/50 extra tests pass on the runner.

Code review of the phase-3 commit found 8 issues, none hitting the open set; all fixed:
- Dynamic smem pool packed like emmy's `pack_smem` (align = max(4, Smem.align)), so the
  layout always fits the `smem_bytes` emmy computed; pool base aligned to the largest
  Smem alignment; pool declared once in `compile_kernel` instead of lazily.
- Seeded loops always start a fresh accumulator register (shadowing outer state, as emmy's
  `T acc = identity;` does), and a loop body is now a scope: names defined inside it are
  dropped after the loop, only that loop's accumulators survive.
- Duplicate Smem name -> error (was silently reused); cp.async nbytes checked (4/8/16).
- cp.async lane offset reuses `flat_index`.

Known gaps for the hidden set: swizzled smem layouts (NotImplementedError), non-f32 dtypes.
Next: more matmul/attention shapes in tests/expressions_extra.py.

Second review (all 5 fixed):
- cp.async lane offset back to emmy's `row * shape[-1] + col` (flat_index was wrong for 3-D slabs).
- Same Smem name with the same shape in sibling scopes reuses the slot (emmy keys Smem by name).
- Seeded accumulators stay writable after their loop; `ctx.assign` copies instead of aliasing
  a register that is updated in place.
- Pool bytes = prod(extents)*4 exactly like pack_smem; no stray blank line before `.extern`.

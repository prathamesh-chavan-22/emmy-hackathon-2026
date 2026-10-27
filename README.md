# Kernel IR → PTX

A compiler from [Emmy](https://github.com/cloudrift-ai/emmy)'s Kernel IR straight to NVIDIA PTX,
written from scratch in Python for the [Emmy Hackathon 2026](docs/HACKATHON.md).

Emmy normally turns a PyTorch model into CUDA C++ and builds it with `nvcc`. This compiler prints
the PTX itself, so only `ptxas` is left, which is about 19× faster. That matters because Emmy's
autotuner builds thousands of kernel variants.

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

Emmy is used only to load the IR (`Graph.from_dict`). Every PTX instruction comes from our own code:
no `nvcc`, no Emmy renderer, no other compiler in the path.

## Results

All runs are on the hosted RTX 5090 runner, checked against torch within each problem's tolerance.

| set | what | pass |
|---|---|---|
| open problems | the 15 in `problems/`, all 5 categories (pointwise → reductions → matmul → fused → attention) | **15/15** |
| extra tests | 175 of our own, generated like the hidden set | **175/175** |
| shape sweep | 43 hidden-set-like sizes | **43/43** |

Speed work: the 512×512 matmul went from 114 to 78 µs, and the 128-token attention from 100 to 54 µs.

## Requirements

- Python 3.12 or newer
- `pip install -r requirements.txt` (installs `emmy-ml==0.3.11`, `numpy`, `requests`)

You need no GPU, CUDA or torch. A hosted runner assembles the PTX with `ptxas`, runs it on a
real GPU and compares the result with the expected output ([docs/API.md](docs/API.md)).

## Quick start

```bash
git clone https://github.com/prathamesh-chavan-22/emmy-hackathon-2026
cd emmy-hackathon-2026
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# compile one kernel and read the PTX
python compile.py problems/p01_vector_add/k_add_pointwise.ir.json

# compile, run on the GPU runner, and check against torch
python submit.py a02_attention_128 --compiler compile.py
#   [k_matmul_softmax_reduce_6f0029] PASS  matmul_1: max abs err 4.172e-07  (43.68 us)
```

`submit.py <problem> [file.ptx | --compiler compile.py]` takes a problem name from `problems/`
(`p00_scalar_add` … `a02_attention_128`).

## How it works

1. **Load:** `compiler/loader.py` reads the IR with Emmy's `Graph.from_dict`, and reads the
   problem's `meta.json` for the parameter order and launch shape.
2. **Translate:** the IR is a tree of typed statements (`Tile`, `Load`, `Accum`, `Loop`, …).
   `compiler/stmts.py` has one handler per statement type, registered with `@handles("Tile")`.
   The walker calls the right handler for each node, and the PTX is the handlers' output joined in order.
3. **Guard:** a `Tile` gives each thread one cell (`ctaid · ntid + tid`). The grid is rounded up
   to whole blocks, so the compiler adds the bounds check the IR leaves out (`setp.ge` + `@p bra`).
4. **Registers:** code uses virtual registers (`%r`, `%rd`, `%f`, `%p`); `ptxas` does the real
   allocation.
5. **Fast paths:** warp-shuffle and shared-memory reductions, `cp.async` multi-stage pipelines,
   `mma.sync.m16n8k16` tensor cores for f16/bf16 with an f32 accumulator, swizzled shared memory,
   and online-softmax attention computed in registers.

[SUBMISSION.md](SUBMISSION.md) has the full feature list (data types, low precision, accuracy,
findings about Emmy). [CHANGELOG.md](CHANGELOG.md) is the step-by-step development log.

## Layout

```
compile.py          command-line entry: IR path in, PTX on stdout
compiler/
  loader.py         Graph.from_dict + meta.json (param order, launch)
  prologue.py       PTX header, signature, ld.param, .maxntid
  stmts.py          statement dispatch table: Tile, Loop, Load/Write, Accum, Cond, ...
  expr.py           index and value expressions, constant folding
  memory.py         addressing, vector loads/stores, swizzle
  coop.py           shared memory, cp.async, barriers, warp shuffles, tree reductions
  mma.py            tensor cores: fragments, mma.sync, ldmatrix, fragment softmax
  ops.py, dtypes.py elementwise op table and per-dtype semantics
  emitter.py        virtual registers and declarations
problems/           the 15 open problems (IR, reference CUDA, meta.json, expected outputs)
tests/              offline checks and GPU test runner, plus our own test problems
tools/              problem generator (torch expression → Emmy → IR)
submit.py           compile + run on the GPU runner + compare
docs/               runner API and the original hackathon brief
```

## Tests

```bash
python tests/check_local.py              # offline: every IR compiles, valid header/entry/params, < 1 s each
python tests/run_all.py                  # the open problems on the GPU runner
python tests/run_all.py --extra          # plus our 175 extra tests
```

Generating new test problems (`tools/generate.py`, `tests/expressions_*.py`) also needs torch:
`pip install -r requirements-dev.txt`. The shape sweep (`--sweep`) is not checked in; regenerate
it from `tests/expressions_sweep.py` first.

## Built with

Written with [Claude Code](https://claude.com/claude-code) as a pair programmer: each feature was
planned in `tasks/todo.md`, built together with its tests, verified on the real GPU, and every
mistake was recorded as a rule in `tasks/lessons.md`.

## License

[MIT](LICENSE). The problem files in `problems/` and the original brief in `docs/HACKATHON.md` come
from the organizers' [task repository](https://github.com/cloudrift-ai/emmy-hackathon-2026).

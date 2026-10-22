#!/usr/bin/env python3
"""Offline sanity gate: compile every IR and check the PTX contract, no GPU needed.

    python tests/check_local.py          # open problems + tests/problems
    python tests/check_local.py r0 x_p   # only problem ids starting with these prefixes

Checks per kernel: compiles without error, pure ASCII, header first, the entry is the
meta.json entry, one .param per arg_order entry, and compile time under a second.
A kernel that hits an unimplemented node is reported as TODO, not as a failure.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = [ROOT / "problems", ROOT / "tests" / "problems"]


def check_kernel(pdir: Path, k: dict) -> tuple[str, str]:
    ir = pdir / f"{k['entry']}.ir.json"
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["?"])[-1]
        return ("TODO" if "NotImplementedError" in last else "FAIL"), last[:90]
    ptx = r.stdout
    problems = []
    if not ptx.isascii():
        problems.append("non-ASCII")
    if not ptx.startswith(".version"):
        problems.append("does not start with .version")
    if not re.search(rf"\.entry\s+{re.escape(k['entry'])}\s*\(", ptx):
        problems.append(f"no .entry {k['entry']}")
    n_params = len(re.findall(r"^\s*\.param\s", ptx, re.M))
    if n_params != len(k["arg_order"]):
        problems.append(f"{n_params} params, arg_order has {len(k['arg_order'])}")
    if dt > 1.0:
        problems.append(f"compile took {dt:.2f}s")
    return ("FAIL", "; ".join(problems)) if problems else ("OK", f"{dt * 1000:.0f} ms")


def main() -> int:
    prefixes = tuple(sys.argv[1:])
    counts = {"OK": 0, "TODO": 0, "FAIL": 0}
    for root in SETS:
        if not root.exists():
            continue
        for pdir in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
            if prefixes and not pdir.name.startswith(prefixes):
                continue
            meta = json.loads((pdir / "meta.json").read_text())
            for k in meta["kernels"]:
                status, detail = check_kernel(pdir, k)
                counts[status] += 1
                print(f"{status:<5} {pdir.name:<24} {k['entry']:<32} {detail}")
    print(f"\n{counts['OK']} ok, {counts['TODO']} todo (unimplemented node), {counts['FAIL']} fail")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())

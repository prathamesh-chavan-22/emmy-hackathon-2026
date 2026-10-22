#!/usr/bin/env python3
"""Run submit.py --compiler compile.py on every open problem and print a table.

    python tests/run_all.py            # all open problems
    python tests/run_all.py p0 r0      # only problems whose id starts with a prefix
    python tests/run_all.py --extra    # also our generated tests in tests/problems
"""

from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def problem_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "meta.json").exists()) if root.exists() else []


def run(pdir: Path) -> tuple[str, str, str, str, str]:
    problem = pdir.name
    r = subprocess.run([sys.executable, str(ROOT / "submit.py"), str(pdir), "--compiler", str(ROOT / "compile.py")],
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    status = "PASS" if r.returncode == 0 else "FAIL"
    regs = ",".join(re.findall(r"(\d+) registers", out)) or "-"
    times = ",".join(re.findall(r"\(([\d.]+) us\)", out)) or "-"
    err = ",".join(re.findall(r"max abs err ([\d.e+-]+)", out)) or "-"
    detail = "" if status == "PASS" else out.strip().splitlines()[-1][:90] if out.strip() else ""
    return problem, status, regs, f"{err} / {times}us", detail


def main() -> int:
    args = sys.argv[1:]
    dirs = problem_dirs(ROOT / "problems")
    if "--extra" in args:
        args.remove("--extra")
        dirs += problem_dirs(ROOT / "tests" / "problems")
    todo = [p for p in dirs if not args or p.name.startswith(tuple(args))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run, todo))
    print(f"{'problem':<20} {'status':<6} {'regs':<8} {'err / time':<28} detail")
    for row in rows:
        print(f"{row[0]:<20} {row[1]:<6} {row[2]:<8} {row[3]:<28} {row[4]}")
    passed = sum(r[1] == "PASS" for r in rows)
    print(f"\n{passed}/{len(rows)} passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

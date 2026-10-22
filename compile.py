#!/usr/bin/env python3
"""Compile one Emmy Kernel IR to PTX.

    python compile.py path/to/kernel.ir.json > out.ptx

Reads meta.json next to the IR for the parameter order and launch geometry.
"""

from __future__ import annotations

import sys

from compiler import compile_kernel, load


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    sys.stdout.write(compile_kernel(load(sys.argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

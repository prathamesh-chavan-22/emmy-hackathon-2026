"""PTX text builder.

Hands out fresh virtual registers and labels, collects instruction lines, and
renders the function body with the register declarations on top. There is no
liveness analysis or register allocation: every value gets its own virtual
register and ptxas does the real allocation downstream.
"""

from __future__ import annotations

import struct

# Register class per PTX type: (name prefix, declaration type). 32-bit integers
# all live in %r (.b32), 64-bit ones and addresses in %rd (.b64), halves in %h
# (.b16); the instruction's own type suffix (.s32, .u64, .f16, ...) decides how
# the bits are interpreted.
REG_CLASSES = {
    "pred": ("%p", ".pred"),
    "f32": ("%f", ".f32"),
    "f64": ("%fd", ".f64"),
    "f16": ("%h", ".b16"),
    "s32": ("%r", ".b32"),
    "u32": ("%r", ".b32"),
    "b32": ("%r", ".b32"),
    "s64": ("%rd", ".b64"),
    "u64": ("%rd", ".b64"),
    "b64": ("%rd", ".b64"),
}


def f32_imm(value: float) -> str:
    """PTX hex float literal (0fXXXXXXXX): exact, unlike a decimal spelling."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(value)))[0]


def f64_imm(value: float) -> str:
    """PTX hex double literal (0dXXXXXXXXXXXXXXXX)."""
    return "0d%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.shared: list[str] = []     # .shared array declarations
        self._counts: dict[str, int] = {}
        self._labels = 0

    # -- names ---------------------------------------------------------------
    def reg(self, kind: str) -> str:
        """A fresh virtual register of the given PTX type."""
        prefix, _ = REG_CLASSES[kind]
        n = self._counts.get(prefix, 0) + 1
        self._counts[prefix] = n
        return f"{prefix}{n}"

    def label(self, hint: str) -> str:
        self._labels += 1
        return f"$L_{hint}_{self._labels}"

    # -- lines ---------------------------------------------------------------
    def emit(self, instr: str, comment: str | None = None) -> None:
        line = f"\t{instr};"
        if comment:
            line += f"\t// {comment}"
        self.lines.append(line)

    def comment(self, text: str) -> None:
        self.lines.append(f"\t// {text}")

    def place(self, label: str) -> None:
        self.lines.append(f"{label}:")

    # -- output --------------------------------------------------------------
    def declare_shared(self, symbol: str, nbytes: int, align: int) -> None:
        self.shared.append(f"\t.shared .align {align} .b8 \t{symbol}[{nbytes}];")

    def declarations(self) -> list[str]:
        """`.reg` lines sized to the highest index handed out per class, then .shared arrays."""
        decls = {}
        for prefix, decl_type in REG_CLASSES.values():
            if prefix in self._counts:
                decls[prefix] = f"\t.reg {decl_type} \t{prefix}<{self._counts[prefix] + 1}>;"
        return list(decls.values()) + self.shared

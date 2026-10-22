"""Per-kernel lowering state shared by the expression and statement lowerers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .emitter import Emitter


@dataclass(frozen=True)
class Value:
    """An operand: a register name or an immediate, plus its PTX type."""
    text: str
    kind: str  # "s32" | "f32" | "pred"
    is_imm: bool = False
    const: object = None  # the Python value when is_imm, for constant folding


@dataclass(frozen=True)
class Buffer:
    """A kernel parameter buffer, already converted to a global address."""
    name: str
    addr: str          # %rd holding the global-space base address
    dtype: str
    shape: tuple[int, ...]

    @property
    def strides(self) -> tuple[int, ...]:
        """Row-major element strides; IR indices are one expression per dim."""
        out, acc = [], 1
        for extent in reversed(self.shape):
            out.append(acc)
            acc *= extent
        return tuple(reversed(out))


@dataclass
class Context:
    em: Emitter
    buffers: dict[str, Buffer]
    threads_launched: int | None          # grid * block, None if unknown
    constants: dict[str, float] = field(default_factory=dict)
    env: dict[str, Value] = field(default_factory=dict)

    def bind(self, name: str, value: Value) -> None:
        self.env[name] = value

    def lookup(self, name: str) -> Value:
        try:
            return self.env[name]
        except KeyError:
            raise KeyError(f"IR name {name!r} used before it is defined") from None

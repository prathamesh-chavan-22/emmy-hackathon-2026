"""Per-kernel lowering state shared by the expression and statement lowerers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from .emitter import Emitter

MOV = {"f32": "mov.f32", "f64": "mov.f64", "s32": "mov.b32", "s64": "mov.b64", "pred": "mov.pred"}


@dataclass(frozen=True)
class Value:
    """An operand: a register name or an immediate, plus its PTX type."""
    text: str
    kind: str  # "f32" | "f64" | "s32" | "s64" | "pred" (see dtypes.py)
    is_imm: bool = False
    const: object = None  # the Python value when is_imm, for constant folding


@dataclass(frozen=True)
class Buffer:
    """A buffer the kernel addresses: a global param or a shared-memory array."""
    name: str
    addr: str          # register with the base address (%rd for global, %r for shared)
    dtype: str
    shape: tuple[int, ...]
    space: str = "global"   # PTX state space: "global" | "shared"

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
    # Names that own a register updated in place (accumulators, carried state).
    # Every other name is SSA: rebinding just points it at a new register.
    mutable: dict[str, Value] = field(default_factory=dict)
    # Names known to equal (register + constant), e.g. an unrolled copy's `i + 2`.
    # Index lowering reads this to put the constant in the address immediate.
    offsets: dict[str, tuple[Value, int]] = field(default_factory=dict)
    # Result register of each `multiply` -> its operand registers, so an Accum
    # can fold `acc + a*b` into one fma (registers are unique: no scoping needed).
    products: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Dynamic shared memory pool (meta smem_bytes > 0): its size and the next free byte.
    smem_pool: int = 0
    smem_used: int = 0
    smem_align: int = 16   # pool base alignment: the largest any array asks for
    # Tensor-core register fragments by name (mma.py); not scoped, as emmy's names are unique.
    frags: dict = field(default_factory=dict)

    def bind(self, name: str, value: Value) -> None:
        self.env[name] = value
        self.offsets.pop(name, None)   # a stale `i + k` form would mis-address

    def lookup(self, name: str) -> Value:
        try:
            return self.env[name]
        except KeyError:
            raise KeyError(f"IR name {name!r} used before it is defined") from None

    def move(self, dst: Value, src: Value) -> None:
        """dst = src, converted to dst's kind (C's assignment conversion: an int
        variable assigned a float truncates it)."""
        if src.kind != dst.kind:
            from .expr import convert   # expr builds on this module, so import late
            src = convert(self, src, dst.kind)
        if dst.kind == "pred" and src.is_imm:
            self.em.emit(f"setp.ne.s32 \t{dst.text}, {int(bool(src.const))}, 0")
        else:
            self.em.emit(f"{MOV[dst.kind]} \t{dst.text}, {src.text}")

    def declare_var(self, name: str, init: Value) -> Value:
        """A fresh register for a mutable name, initialized to `init`."""
        var = Value(self.em.reg(init.kind), init.kind)
        self.move(var, init)
        self.mutable[name] = var
        self.env[name] = var
        return var

    @contextmanager
    def scope(self, release=()):
        """A nested block: names bound inside it are dropped when it ends.

        `release` names stop being mutable inside the block, so assignments
        there rebind them (scratch copies) instead of overwriting the outer
        register; the outer binding is back when the block ends.
        """
        outer = dict(self.env), dict(self.mutable), dict(self.offsets)
        for name in release:
            self.mutable.pop(name, None)
        try:
            yield
        finally:
            self.env, self.mutable, self.offsets = outer

    def assign(self, name: str, value: Value) -> None:
        """Set `name`: write into its own register if it is mutable, else rebind.

        Loop-carried values must stay in one register across iterations, so a
        mutable name is never re-pointed; ptxas coalesces the extra mov.
        """
        var = self.mutable.get(name)
        if var is None:
            if any(value.text == m.text for m in self.mutable.values()):
                # Snapshot: aliasing a register that is updated in place would let
                # this name see later updates.
                copy = Value(self.em.reg(value.kind), value.kind)
                self.move(copy, value)
                value = copy
            self.env[name] = value
            self.offsets.pop(name, None)
        elif value.text != var.text:
            self.move(var, value)

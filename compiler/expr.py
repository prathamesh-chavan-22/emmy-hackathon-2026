"""Recursive lowering of IR expressions (Var / Literal / BinaryExpr / TernaryExpr).

Expressions are small index and predicate trees. Each lowers to a Value: an
immediate when it is a compile-time constant (all problems have static shapes,
so most index arithmetic folds away), otherwise a fresh register.
"""

from __future__ import annotations

import operator

from .context import Context, Value
from .dtypes import DType, INT_KINDS, bits, cast_const, common_kind, cvt
from .emitter import f32_imm, f64_imm

_FOLD = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
    "&&": lambda a, b: bool(a and b), "||": lambda a, b: bool(a or b),
    "^": operator.xor,
}

_INT_OPS = {"+": "add", "-": "sub", "*": "mul.lo", "/": "div", "//": "div", "%": "rem", "^": "xor"}
_F32_OPS = {"+": "add.f32", "-": "sub.f32", "*": "mul.f32", "/": "div.rn.f32"}
_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_LOGIC = {"&&": "and.pred", "||": "or.pred", "^": "xor.pred"}


def imm(value, kind: str | None = None) -> Value:
    """A constant operand; `kind` defaults to the Python type's (bool, int, float)."""
    if kind is None:
        kind = "pred" if isinstance(value, bool) else "s32" if isinstance(value, int) else "f32"
    v = cast_const(value, kind)
    if kind == "pred":
        return Value("1" if v else "0", kind, True, v)
    if kind in INT_KINDS:
        return Value(str(v), kind, True, v)
    if kind == "f64":
        return Value(f64_imm(v), kind, True, v)
    return Value(f32_imm(v), "f32", True, v)


def const(value, dt: DType) -> Value:
    """A constant of IR dtype `dt`: an f16/bf16/fp8 constant is rounded to that format."""
    return imm(cast_const(value, dt.kind, dt.mem if dt.rounds else None), dt.kind)


def _fold(op: str, a, b):
    """Constant-fold integer/float arithmetic. IR int division truncates (C semantics)."""
    if op in ("/", "//") and isinstance(a, int) and isinstance(b, int):
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q
    if op == "%" and isinstance(a, int) and isinstance(b, int):
        return a - b * _fold("/", a, b)
    if op == "/":
        return a / b
    return _FOLD[op](a, b)


def to_f32(ctx: Context, v: Value) -> Value:
    return convert(ctx, v, "f32")


def convert(ctx: Context, v: Value, kind: str) -> Value:
    """`v` as a value of `kind`, with C's implicit conversion rules.

    A number is true when nonzero (NaN included, hence the unordered compare);
    a bool is 1 or 0; a float becomes an int by truncation. emmy's C leans on
    these implicit casts for masks and for int variables fed float math.
    """
    if v.kind == kind:
        return v
    if v.is_imm:
        return imm(v.const, kind)
    em = ctx.em
    r = em.reg(kind)
    if v.kind == "pred":
        em.emit(f"selp.{kind} \t{r}, {imm(1, kind).text}, {imm(0, kind).text}, {v.text}", "bool -> 1/0")
    elif kind == "pred":
        test = "ne" if v.kind in INT_KINDS else "neu"
        em.emit(f"setp.{test}.{v.kind} \t{r}, {v.text}, {imm(0, v.kind).text}", "nonzero -> true")
    else:
        cvt(em, r, kind, v.text, v.kind)
    return Value(r, kind)


def to_reg(ctx: Context, v: Value) -> Value:
    """Materialize an immediate in a register (for instructions that need one)."""
    if not v.is_imm:
        return v
    r = ctx.em.reg(v.kind)
    if v.kind == "pred":
        ctx.em.emit(f"setp.ne.s32 \t{r}, {v.text}, 0")
    else:
        ctx.em.emit(f"mov.{v.kind if v.kind in ('f32', 'f64') else f'b{bits(v.kind)}'} \t{r}, {v.text}")
    return Value(r, v.kind)


def lower(ctx: Context, e) -> Value:
    kind = type(e).__name__
    if kind == "Var":
        return ctx.lookup(e.name)
    if kind == "Literal":
        return imm(e.value)
    if kind == "BinaryExpr":
        return binary(ctx, e.op, lower(ctx, e.left), lower(ctx, e.right))
    if kind == "TernaryExpr":
        return select(ctx, lower(ctx, e.cond), lower(ctx, e.if_true), lower(ctx, e.if_false))
    raise NotImplementedError(f"expression {kind}: {e}")


def binary(ctx: Context, op: str, a: Value, b: Value) -> Value:
    """`a op b` on lowered operands: folded if both are constant, else one instruction.

    Also the building block for index math the statement lowerers compose themselves.
    """
    em = ctx.em
    if a.is_imm and b.is_imm:
        return imm(_fold(op, a.const, b.const))
    # Cheap identities keep unrolled index math (a0*2 + 0) down to one instruction.
    if op in ("+", "-") and b.is_imm and b.const == 0:
        return a
    if op == "+" and a.is_imm and a.const == 0:
        return b
    if op == "*" and b.is_imm and b.const == 1:
        return a
    if op == "*" and a.is_imm and a.const == 1:
        return b

    if op in _LOGIC:
        a, b = to_reg(ctx, a), to_reg(ctx, b)
        r = em.reg("pred")
        em.emit(f"{_LOGIC[op]} \t{r}, {a.text}, {b.text}")
        return Value(r, "pred")

    kind = common_kind((a.kind, b.kind))
    a, b = convert(ctx, a, kind), convert(ctx, b, kind)
    if op in _CMP:
        r = em.reg("pred")
        em.emit(f"setp.{_CMP[op]}.{kind} \t{r}, {to_reg(ctx, a).text}, {b.text}")
        return Value(r, "pred")
    if kind in ("f32", "f64"):
        instr = _F32_OPS[op].replace("f32", kind)
    else:
        instr = f"{_INT_OPS[op]}.{f'b{bits(kind)}' if op == '^' else kind}"
    r = em.reg(kind)
    em.emit(f"{instr} \t{r}, {to_reg(ctx, a).text}, {b.text}")
    return Value(r, kind)


def select(ctx: Context, cond: Value, t: Value, f: Value) -> Value:
    if cond.is_imm:
        return t if cond.const else f
    cond = convert(ctx, cond, "pred")
    kind = common_kind((t.kind, f.kind))
    t, f = convert(ctx, t, kind), convert(ctx, f, kind)
    r = ctx.em.reg(kind)
    ctx.em.emit(f"selp.{kind} \t{r}, {t.text}, {f.text}, {cond.text}")
    return Value(r, kind)

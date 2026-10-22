"""Recursive lowering of IR expressions (Var / Literal / BinaryExpr / TernaryExpr).

Expressions are small index and predicate trees. Each lowers to a Value: an
immediate when it is a compile-time constant (all problems have static shapes,
so most index arithmetic folds away), otherwise a fresh register.
"""

from __future__ import annotations

import operator

from .context import Context, Value
from .emitter import f32_imm

_FOLD = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
    "&&": lambda a, b: bool(a and b), "||": lambda a, b: bool(a or b),
    "^": operator.xor,
}

_INT_OPS = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32",
            "/": "div.s32", "//": "div.s32", "%": "rem.s32", "^": "xor.b32"}
_F32_OPS = {"+": "add.f32", "-": "sub.f32", "*": "mul.f32", "/": "div.rn.f32"}
_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_LOGIC = {"&&": "and.pred", "||": "or.pred", "^": "xor.pred"}


def imm(value) -> Value:
    if isinstance(value, bool):
        return Value("1" if value else "0", "pred", True, value)
    if isinstance(value, int):
        return Value(str(value), "s32", True, value)
    return Value(f32_imm(value), "f32", True, float(value))


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
    if v.kind == "f32":
        return v
    if v.is_imm:
        return imm(float(v.const))
    r = ctx.em.reg("f32")
    ctx.em.emit(f"cvt.rn.f32.s32 \t{r}, {v.text}")
    return Value(r, "f32")


def to_reg(ctx: Context, v: Value) -> Value:
    """Materialize an immediate in a register (for instructions that need one)."""
    if not v.is_imm:
        return v
    r = ctx.em.reg(v.kind)
    ctx.em.emit(f"mov.{'b32' if v.kind == 's32' else v.kind} \t{r}, {v.text}"
                if v.kind != "pred" else f"setp.ne.s32 \t{r}, {v.text}, 0")
    return Value(r, v.kind)


def lower(ctx: Context, e) -> Value:
    kind = type(e).__name__
    if kind == "Var":
        return ctx.lookup(e.name)
    if kind == "Literal":
        return imm(e.value)
    if kind == "BinaryExpr":
        return _binary(ctx, e.op, lower(ctx, e.left), lower(ctx, e.right))
    if kind == "TernaryExpr":
        return _select(ctx, lower(ctx, e.cond), lower(ctx, e.if_true), lower(ctx, e.if_false))
    raise NotImplementedError(f"expression {kind}: {e}")


def _binary(ctx: Context, op: str, a: Value, b: Value) -> Value:
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

    is_float = "f32" in (a.kind, b.kind)
    if is_float:
        a, b = to_f32(ctx, a), to_f32(ctx, b)
    if op in _CMP:
        r = em.reg("pred")
        em.emit(f"setp.{_CMP[op]}.{'f32' if is_float else 's32'} \t{r}, {to_reg(ctx, a).text}, {b.text}")
        return Value(r, "pred")
    table = _F32_OPS if is_float else _INT_OPS
    r = em.reg("f32" if is_float else "s32")
    em.emit(f"{table[op]} \t{r}, {to_reg(ctx, a).text}, {b.text}")
    return Value(r, "f32" if is_float else "s32")


def _select(ctx: Context, cond: Value, t: Value, f: Value) -> Value:
    if cond.is_imm:
        return t if cond.const else f
    if "f32" in (t.kind, f.kind):
        t, f = to_f32(ctx, t), to_f32(ctx, f)
    kind = t.kind
    r = ctx.em.reg(kind)
    ctx.em.emit(f"selp.{'b32' if kind == 's32' else kind} \t{r}, {t.text}, {f.text}, {cond.text}")
    return Value(r, kind)

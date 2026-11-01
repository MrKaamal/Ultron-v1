"""Safe arithmetic evaluator (AST based; no eval)."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from ultron.errors import ToolError
from ultron.tools.base import Risk, Tool, ToolContext

MAX_EXPR_LEN = 200
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e300

_BIN: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > MAX_EXPONENT or abs(left) > MAX_MAGNITUDE):
            raise ToolError("exponent too large")
        result = _BIN[type(node.op)](left, right)
        if isinstance(result, (int, float)) and abs(result) > MAX_MAGNITUDE:
            raise ToolError("result too large")
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def evaluate(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        raise ToolError("empty expression")
    if len(expression) > MAX_EXPR_LEN:
        raise ToolError(f"expression too long (max {MAX_EXPR_LEN} chars)")
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval(tree)
    except ToolError:
        raise
    except SyntaxError as exc:
        raise ToolError(f"syntax error in expression: {exc.msg}") from None
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except (ValueError, TypeError, OverflowError) as exc:
        raise ToolError(f"cannot evaluate: {exc}") from None
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return str(value)


def _calc(ctx: ToolContext, args: dict[str, Any]) -> str:
    return evaluate(str(args["expression"]))


TOOLS: list[Tool] = [
    Tool(
        name="calc",
        description=(
            "Evaluate an arithmetic expression exactly. Supports + - * / // % **, parentheses, "
            "sqrt sin cos tan log log10 exp abs round min max floor ceil, and the constants pi and e. "
            "Use this instead of doing arithmetic in your head."
        ),
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "For example '(17 * 23) + 4'."}},
            "required": ["expression"],
        },
        func=_calc,
        risk=Risk.SAFE,
    ),
]

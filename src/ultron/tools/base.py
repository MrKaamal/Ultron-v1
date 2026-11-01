"""Tool definition and argument coercion. (The registry itself lives in ``ultron.tools``.)"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ultron.messages import ToolSpec

if TYPE_CHECKING:
    from ultron.config import Config
    from ultron.memory import Memory
    from ultron.sandbox import Sandbox


class Risk(StrEnum):
    SAFE = "safe"  # runs without asking
    CONFIRM = "confirm"  # needs approval (human, or config auto_approve)
    FORBIDDEN = "forbidden"  # never runs


class Verdict(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""

    @classmethod
    def allow(cls, reason: str = "") -> Decision:
        return cls(Verdict.ALLOW, reason)

    @classmethod
    def ask(cls, reason: str = "") -> Decision:
        return cls(Verdict.ASK, reason)

    @classmethod
    def deny(cls, reason: str = "") -> Decision:
        return cls(Verdict.DENY, reason)


@dataclass
class ToolContext:
    cfg: Config
    sandbox: Sandbox
    memory: Memory
    run_id: str = ""


ToolFunc = Callable[[ToolContext, dict[str, Any]], str]
CheckFunc = Callable[[ToolContext, dict[str, Any]], "Decision | None"]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc
    risk: Risk = Risk.CONFIRM  # fail-safe default: an unlabelled tool asks first
    check: CheckFunc | None = None
    source: str = "builtin"

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


def tool(
    *,
    description: str,
    name: str | None = None,
    parameters: dict[str, Any] | None = None,
    risk: str | Risk = Risk.CONFIRM,
    check: CheckFunc | None = None,
) -> Callable[[ToolFunc], ToolFunc]:
    """Decorator that marks a function as an ULTRON tool (used by skills and built-ins).

    The function receives ``(ctx, args)`` and returns a string for the model.
    """

    def deco(fn: ToolFunc) -> ToolFunc:
        fn.__ultron_tool__ = Tool(  # type: ignore[attr-defined]
            name=name or fn.__name__,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            func=fn,
            risk=Risk(risk),
            check=check,
        )
        return fn

    return deco


def _coerce(value: Any, spec: dict[str, Any]) -> tuple[Any, str]:
    kind = spec.get("type")
    if kind == "string":
        if isinstance(value, str):
            out: Any = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out = str(value)
        else:
            return None, "expected a string"
    elif kind == "integer":
        if isinstance(value, bool):
            return None, "expected an integer"
        if isinstance(value, float) and value.is_integer():
            out = int(value)
        elif isinstance(value, int):
            out = value
        elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
            out = int(value)
        else:
            return None, "expected an integer"
    elif kind == "number":
        try:
            out = float(value) if not isinstance(value, bool) else None
        except (TypeError, ValueError):
            out = None
        if out is None:
            return None, "expected a number"
    elif kind == "boolean":
        if isinstance(value, bool):
            out = value
        elif isinstance(value, str) and value.lower() in ("true", "false"):
            out = value.lower() == "true"
        else:
            return None, "expected true or false"
    elif kind == "array":
        if not isinstance(value, list):
            return None, "expected an array"
        out = value
    elif kind == "object":
        if not isinstance(value, dict):
            return None, "expected an object"
        out = value
    else:
        out = value
    if "enum" in spec and out not in spec["enum"]:
        return None, f"must be one of {spec['enum']}"
    return out, ""


def coerce_args(schema: dict[str, Any], args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce obvious type slips (``"5"`` -> 5), drop unknown extras, report schema problems."""
    props = schema.get("properties", {})
    problems = [
        f"missing required argument '{k}'" for k in schema.get("required", []) if k not in args
    ]
    clean: dict[str, Any] = {}
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            continue  # ignore unknown extras
        out, err = _coerce(value, spec)
        if err:
            problems.append(f"argument '{key}': {err}")
        else:
            clean[key] = out
    return clean, problems

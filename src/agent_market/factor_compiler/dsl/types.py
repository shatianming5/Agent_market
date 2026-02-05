from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent_market.factor_compiler.api_models import ExprArg, ExprNode
from agent_market.factor_compiler.dsl.grammar import CONST_OP, VAR_OP


@dataclass(frozen=True, slots=True)
class FactorType:
    """
    Minimal type descriptor for Factor Compiler expressions.

    This is intentionally lightweight: it provides enough structure to support
    basic typechecking and to carry the plan.md-required attributes, without
    enforcing a full-blown type system yet.
    """

    kind: str  # scalar|series|frame|event_stream|lob_state|lob_series
    dtype: str = "float"  # float|int|bool|unknown
    semantic: Optional[str] = None  # price|volume|return|...
    freq: Optional[str] = None
    timezone: str = "UTC"
    availability_delay_ms: int = 0
    lookback: int = 0

    def is_scalar(self) -> bool:
        return self.kind == "scalar"

    def is_series(self) -> bool:
        return self.kind == "series"


SCALAR_FLOAT = FactorType(kind="scalar", dtype="float")
SCALAR_INT = FactorType(kind="scalar", dtype="int")
SCALAR_BOOL = FactorType(kind="scalar", dtype="bool")

SERIES_FLOAT = FactorType(kind="series", dtype="float")
SERIES_BOOL = FactorType(kind="series", dtype="bool")


def _literal_type(value: Any) -> FactorType:
    if isinstance(value, bool):
        return SCALAR_BOOL
    if isinstance(value, int):
        return SCALAR_INT
    if isinstance(value, float):
        return SCALAR_FLOAT
    return FactorType(kind="scalar", dtype="unknown")


def unify(a: FactorType, b: FactorType) -> FactorType:
    """
    Best-effort unification for binary operators.

    Rules:
      - series dominates scalar
      - bool dominates unknown only for comparisons
      - dtype is promoted to float when mixing int/float
      - lookback is max
    """

    kind = "series" if (a.kind == "series" or b.kind == "series") else "scalar"

    def _dtype(x: str) -> str:
        return x or "unknown"

    da = _dtype(a.dtype)
    db = _dtype(b.dtype)
    if "unknown" in (da, db):
        dtype = "unknown" if (da == "unknown" and db == "unknown") else ("float" if "float" in (da, db) else "unknown")
    elif "float" in (da, db):
        dtype = "float"
    elif "int" in (da, db):
        dtype = "int"
    else:
        dtype = da

    return FactorType(
        kind=kind,
        dtype=dtype,
        semantic=a.semantic or b.semantic,
        freq=a.freq or b.freq,
        timezone=a.timezone or b.timezone,
        availability_delay_ms=max(int(a.availability_delay_ms), int(b.availability_delay_ms)),
        lookback=max(int(a.lookback), int(b.lookback)),
    )


_CALL_RETURNS: Dict[str, FactorType] = {
    # time-series transforms
    "shift": SERIES_FLOAT,
    "lag": SERIES_FLOAT,
    "diff": SERIES_FLOAT,
    "pct_change": SERIES_FLOAT,
    "roll_mean": SERIES_FLOAT,
    "roll_std": SERIES_FLOAT,
    "rolling_sum": SERIES_FLOAT,
    "rolling_min": SERIES_FLOAT,
    "rolling_max": SERIES_FLOAT,
    "ema": SERIES_FLOAT,
    "z": SERIES_FLOAT,
    "ts_z": SERIES_FLOAT,
    "ifelse": SERIES_FLOAT,
    # math
    "abs": SERIES_FLOAT,
    "log": SERIES_FLOAT,
    "log1p": SERIES_FLOAT,
    "exp": SERIES_FLOAT,
    "sqrt": SERIES_FLOAT,
    "tanh": SERIES_FLOAT,
    "clip": SERIES_FLOAT,
    "sign": SERIES_FLOAT,
}


def infer_expr_type(expr: ExprNode, *, var_types: Optional[Dict[str, FactorType]] = None) -> FactorType:
    """
    Best-effort type inference for an ExprNode.

    This is designed to be *non-blocking*: unknown operators default to Series[float].
    """

    var_types = dict(var_types or {})

    def _infer(arg: ExprArg) -> FactorType:
        if isinstance(arg, ExprNode):
            op = str(arg.op)
            args = list(arg.args or [])

            if op == VAR_OP:
                name = str(args[0]) if args else ""
                return var_types.get(name, SERIES_FLOAT)
            if op == CONST_OP:
                return _literal_type(args[0] if args else None)

            if op in {"pos", "neg"}:
                return _infer(args[0]) if args else SERIES_FLOAT
            if op in {"add", "sub", "mul", "div", "pow", "mod", "floordiv"}:
                if len(args) != 2:
                    return SERIES_FLOAT
                return unify(_infer(args[0]), _infer(args[1]))
            if op in {"gt", "gte", "lt", "lte", "eq", "neq"}:
                return SERIES_BOOL if (any(_infer(a).is_series() for a in args)) else SCALAR_BOOL

            # function-style operators
            base = _CALL_RETURNS.get(op)
            if base is not None:
                # propagate time attributes from the first series arg (if any)
                series_arg = next((a for a in args if isinstance(a, ExprNode) or not isinstance(a, (int, float))), None)
                inferred = _infer(series_arg) if series_arg is not None else base
                return FactorType(
                    kind=inferred.kind,
                    dtype=base.dtype,
                    semantic=inferred.semantic,
                    freq=inferred.freq,
                    timezone=inferred.timezone,
                    availability_delay_ms=inferred.availability_delay_ms,
                    lookback=inferred.lookback,
                )

            return SERIES_FLOAT

        return _literal_type(arg)

    return _infer(expr)


__all__ = [
    "FactorType",
    "SCALAR_BOOL",
    "SCALAR_FLOAT",
    "SCALAR_INT",
    "SERIES_BOOL",
    "SERIES_FLOAT",
    "infer_expr_type",
    "unify",
]


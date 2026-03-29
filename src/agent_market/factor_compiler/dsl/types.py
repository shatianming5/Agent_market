from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from agent_market.factor_compiler.api_models import ExprArg, ExprNode
from agent_market.factor_compiler.dsl.grammar import CONST_OP, VAR_OP


# ---------------------------------------------------------------------------
# Enums for Kind, Dtype, Semantic — plan.md §3.3.1 "核心类型"
# ---------------------------------------------------------------------------

class Kind(str, Enum):
    """Expression kind — determines how values are shaped."""
    SCALAR = "scalar"
    SERIES = "series"
    FRAME = "frame"
    EVENT_STREAM = "event_stream"
    LOB_STATE = "lob_state"
    LOB_SERIES = "lob_series"


class Dtype(str, Enum):
    """Primitive data type carried by an expression."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    UNKNOWN = "unknown"


class Semantic(str, Enum):
    """Semantic tag — declares what the value *means* for domain rules."""
    PRICE = "price"
    VOLUME = "volume"
    RETURN = "return"
    RATIO = "ratio"
    COUNT = "count"
    DURATION = "duration"
    INDICATOR = "indicator"
    COST = "cost"
    PROBABILITY = "probability"
    SIGNAL = "signal"


# ---------------------------------------------------------------------------
# FactorType — plan.md §3.3.2 "类型属性（必须携带）"
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FactorType:
    """
    Type descriptor for Factor Compiler expressions.

    Every expression infers a FactorType that carries both structural
    information (kind, dtype) and domain metadata (semantic, freq,
    timezone, availability_delay_ms, lookback).  The ``validate()``
    method enforces that required attributes are present and within
    acceptable ranges — forming the "强制携带 + 强约束校验" loop
    required by plan.md §3.3.2.
    """

    kind: str  # value from Kind enum
    dtype: str = "float"  # value from Dtype enum
    semantic: Optional[str] = None  # value from Semantic enum (optional)
    freq: Optional[str] = None
    timezone: str = "UTC"
    availability_delay_ms: int = 0
    lookback: int = 0

    def is_scalar(self) -> bool:
        return self.kind == Kind.SCALAR or self.kind == "scalar"

    def is_series(self) -> bool:
        return self.kind == Kind.SERIES or self.kind == "series"

    # --- plan.md §3.3.2: 强制携带 + 强约束校验 ---

    def validate(self, *, max_lookback: int = 2000, min_delay_ms: int = 0) -> List[str]:
        """Return a list of validation error strings (empty == valid).

        Enforces:
          - kind must be a recognised Kind value
          - dtype must be a recognised Dtype value
          - semantic (if set) must be a recognised Semantic value
          - availability_delay_ms >= 0
          - lookback >= 0 and <= max_lookback
          - availability_delay_ms <= min_delay_ms (tradability gate)
        """
        errors: List[str] = []
        valid_kinds = {k.value for k in Kind}
        valid_dtypes = {d.value for d in Dtype}
        valid_semantics = {s.value for s in Semantic}

        if self.kind not in valid_kinds:
            errors.append(f"invalid kind={self.kind!r}; expected one of {sorted(valid_kinds)}")
        if self.dtype not in valid_dtypes:
            errors.append(f"invalid dtype={self.dtype!r}; expected one of {sorted(valid_dtypes)}")
        if self.semantic is not None and self.semantic not in valid_semantics:
            errors.append(f"invalid semantic={self.semantic!r}; expected one of {sorted(valid_semantics)}")
        if self.availability_delay_ms < 0:
            errors.append(f"availability_delay_ms={self.availability_delay_ms} must be >= 0")
        if self.lookback < 0:
            errors.append(f"lookback={self.lookback} must be >= 0")
        if self.lookback > max_lookback:
            errors.append(f"lookback={self.lookback} exceeds max_lookback={max_lookback}")
        if min_delay_ms > 0 and self.availability_delay_ms > min_delay_ms:
            errors.append(
                f"availability_delay_ms={self.availability_delay_ms} "
                f"exceeds tradability gate min_delay_ms={min_delay_ms}"
            )
        return errors


SCALAR_FLOAT = FactorType(kind="scalar", dtype="float")
SCALAR_INT = FactorType(kind="scalar", dtype="int")
SCALAR_BOOL = FactorType(kind="scalar", dtype="bool")

SERIES_FLOAT = FactorType(kind="series", dtype="float")
SERIES_BOOL = FactorType(kind="series", dtype="bool")

# Domain-specific presets
SERIES_PRICE = FactorType(kind="series", dtype="float", semantic="price")
SERIES_VOLUME = FactorType(kind="series", dtype="float", semantic="volume")
SERIES_RETURN = FactorType(kind="series", dtype="float", semantic="return")


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
    "diff": SERIES_RETURN,
    "pct_change": SERIES_RETURN,
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
    # microstructure (semantic-aware)
    "mid": SERIES_PRICE,
    "spread": SERIES_PRICE,
    "microprice": SERIES_PRICE,
    "vwap": SERIES_PRICE,
    "depth_bid": SERIES_VOLUME,
    "depth_ask": SERIES_VOLUME,
    "imbalance": SERIES_FLOAT,
    "ofi": SERIES_VOLUME,
    "rv": SERIES_RETURN,
}

# Operators whose first numeric arg is a window/lookback length.
# Used by infer_expr_type to propagate lookback correctly.
_WINDOW_OPS = frozenset({
    "roll_mean", "roll_std", "rolling_sum", "rolling_min", "rolling_max",
    "ema", "ts_z", "z",
})


def infer_expr_type(expr: ExprNode, *, var_types: Optional[Dict[str, FactorType]] = None) -> FactorType:
    """
    Best-effort type inference for an ExprNode.

    This is designed to be *non-blocking*: unknown operators default to Series[float].
    Propagates lookback from window-based operators and semantic tags from the
    operator registry.
    """

    var_types = dict(var_types or {})

    def _extract_window(op: str, args: list) -> int:
        """Extract the window/lookback arg for rolling operators."""
        if op not in _WINDOW_OPS:
            return 0
        for a in args:
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                return max(0, int(a))
        return 0

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
                series_arg = next(
                    (a for a in args if isinstance(a, ExprNode) or not isinstance(a, (int, float))),
                    None,
                )
                inferred = _infer(series_arg) if series_arg is not None else base
                window = _extract_window(op, args)
                return FactorType(
                    kind=inferred.kind,
                    dtype=base.dtype,
                    semantic=base.semantic or inferred.semantic,
                    freq=inferred.freq,
                    timezone=inferred.timezone,
                    availability_delay_ms=inferred.availability_delay_ms,
                    lookback=max(inferred.lookback, window),
                )

            return SERIES_FLOAT

        return _literal_type(arg)

    return _infer(expr)


# ---------------------------------------------------------------------------
# typecheck — plan.md §3.5.2 统一类型检查入口
# ---------------------------------------------------------------------------

def typecheck(
    expr: ExprNode,
    *,
    var_types: Optional[Dict[str, FactorType]] = None,
    max_lookback: int = 2000,
    min_delay_ms: int = 0,
) -> "TypecheckResult":
    """Unified typecheck entry point (plan.md §3.5.2).

    Performs type inference, then validates all structural and domain
    constraints on the resulting FactorType.  Returns a ``TypecheckResult``
    containing the inferred type and any validation errors.
    """
    inferred = infer_expr_type(expr, var_types=var_types)
    errors = inferred.validate(max_lookback=max_lookback, min_delay_ms=min_delay_ms)
    return TypecheckResult(inferred_type=inferred, errors=errors)


@dataclass
class TypecheckResult:
    """Result of ``typecheck()``."""
    inferred_type: FactorType
    errors: List[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


__all__ = [
    "Dtype",
    "FactorType",
    "Kind",
    "SCALAR_BOOL",
    "SCALAR_FLOAT",
    "SCALAR_INT",
    "SERIES_BOOL",
    "SERIES_FLOAT",
    "SERIES_PRICE",
    "SERIES_RETURN",
    "SERIES_VOLUME",
    "Semantic",
    "TypecheckResult",
    "infer_expr_type",
    "typecheck",
    "unify",
]


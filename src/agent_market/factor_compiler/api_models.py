from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


ExprArg = Union["ExprNode", int, float, str, bool, None]


class ExprNode(BaseModel):
    """Canonical AST node for Factor Compiler DSL."""

    op: str = Field(..., description="Operator name, e.g. zscore, roll_mean")
    args: List[ExprArg] = Field(default_factory=list, description="Operator arguments")

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json") if hasattr(self, "model_dump") else self.dict()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class ComplexityBudget(BaseModel):
    max_nodes: int = Field(40, ge=1)
    max_depth: int = Field(8, ge=1)
    max_expensive_ops: int = Field(12, ge=0)


class FactorConstraints(BaseModel):
    max_lookback: int = Field(2000, ge=0)
    min_delay_ms: int = Field(0, ge=0)
    max_turnover: float = Field(8.0, ge=0.0)
    complexity_budget: ComplexityBudget = Field(default_factory=ComplexityBudget)


class FactorTest(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Test type, e.g. no_lookahead, nan_rate")
    horizon: Optional[int] = Field(None, description="Optional horizon for time-safety tests")
    max_nan_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)


class FactorMeta(BaseModel):
    timeframe: str = Field(..., description="Sampling timeframe, e.g. 1s, 1m, 1h")
    universe: List[str] = Field(default_factory=list, description="Pairs/symbols, e.g. BTC/USDT")
    data_sources: List[str] = Field(default_factory=list, description="e.g. trades, lob_l2, ohlcv")


class FactorSpec(BaseModel):
    """LLM/authoritative input format for Factor Compiler."""

    name: str
    version: str = Field("1.0")
    hypothesis: Optional[str] = None
    expr: ExprNode
    constraints: FactorConstraints = Field(default_factory=FactorConstraints)
    tests: List[FactorTest] = Field(default_factory=list)
    meta: FactorMeta

    def canonical_expr_json(self) -> str:
        return self.expr.canonical_json()

    def canonical_expr_sha256(self) -> str:
        return self.expr.canonical_sha256()

    def canonical_spec_json(self) -> str:
        payload = self.model_dump(mode="json") if hasattr(self, "model_dump") else self.dict()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def canonical_spec_sha256(self) -> str:
        return hashlib.sha256(self.canonical_spec_json().encode("utf-8")).hexdigest()


def factor_spec_json_schema() -> Dict[str, Any]:
    try:
        return FactorSpec.model_json_schema()  # pydantic v2
    except AttributeError:  # pragma: no cover
        return FactorSpec.schema()  # pydantic v1


def write_factor_spec_schema(path: Path) -> Path:
    """Write FactorSpec JSON schema to disk (for API and tooling)."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(factor_spec_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


# Rebuild forward refs for pydantic recursion support
try:
    ExprNode.model_rebuild()  # pydantic v2
except AttributeError:  # pragma: no cover
    ExprNode.update_forward_refs()  # pydantic v1


__all__ = [
    "ComplexityBudget",
    "ExprArg",
    "ExprNode",
    "FactorConstraints",
    "FactorMeta",
    "FactorSpec",
    "FactorTest",
    "factor_spec_json_schema",
    "write_factor_spec_schema",
]

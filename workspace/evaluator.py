"""Multi-objective evaluator — scores backtest results against objectives.json.

Usage:
    from workspace.evaluator import evaluate
    score = evaluate(backtest_result)
    # score = {"total_score": 72.5, "grade": "B", "details": {...}, "suggestions": [...]}

Accepts both Dict[str, Any] and workspace.types.BacktestResult.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workspace.types import BacktestResult
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES_PATH = ROOT / "workspace" / "objectives.json"


def _load_objectives() -> Dict[str, Any]:
    return json.loads(OBJECTIVES_PATH.read_text(encoding="utf-8"))


def _score_target(value: float, target: Dict[str, Any]) -> float:
    """Score a single target on 0-100 scale."""
    direction = target.get("direction", "maximize")

    if direction == "maximize":
        threshold = target.get("min", 0)
        if threshold == 0:
            return min(100.0, max(0.0, value * 100))
        if value >= threshold:
            # Above target: 70-100 based on how much above
            excess = (value - threshold) / max(abs(threshold), 0.01)
            return min(100.0, 70.0 + 30.0 * min(excess, 1.0))
        else:
            # Below target: 0-70 based on how close
            ratio = value / threshold if threshold != 0 else 0
            return max(0.0, 70.0 * max(ratio, 0.0))
    else:  # minimize
        threshold = target.get("max", 100)
        if value <= threshold:
            # Within target: 70-100
            ratio = value / threshold if threshold != 0 else 0
            return min(100.0, 70.0 + 30.0 * (1.0 - min(ratio, 1.0)))
        else:
            # Above target: 0-70
            excess = (value - threshold) / max(abs(threshold), 0.01)
            return max(0.0, 70.0 * max(1.0 - excess, 0.0))


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C+"
    if score >= 50:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _metric_key_map() -> Dict[str, str]:
    """Map objectives.json target names → backtest result keys."""
    return {
        "sharpe": "sharpe",
        "sortino": "sortino",
        "max_drawdown_pct": "max_drawdown_pct",
        "profit_pct": "profit_pct",
        "win_rate": "win_rate",
        "profit_factor": "profit_factor",
    }


def evaluate(
    result: "Dict[str, Any] | BacktestResult",
    objectives_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a backtest result against multi-objective targets.

    Args:
        result: Backtest result dict from run_backtest() (must have ok=True).
        objectives_path: Override path to objectives.json.

    Returns:
        {
            "total_score": float (0-100),
            "grade": str (A/B+/B/C+/C/D/F),
            "details": {target_name: {value, target, score, met}},
            "constraint_violations": [str],
            "suggestions": [str],
        }
    """
    if not result.get("ok"):
        return {
            "total_score": 0,
            "grade": "F",
            "details": {},
            "constraint_violations": ["backtest failed"],
            "suggestions": ["fix strategy code errors first"],
        }

    obj = _load_objectives() if objectives_path is None else json.loads(Path(objectives_path).read_text())
    targets = obj.get("targets", {})
    constraints = obj.get("constraints", {})
    key_map = _metric_key_map()

    details: Dict[str, Any] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for target_name, target_def in targets.items():
        result_key = key_map.get(target_name, target_name)
        value = result.get(result_key)
        if value is None:
            details[target_name] = {"value": None, "target": target_def, "score": 0, "met": False}
            continue

        value = float(value)
        score = _score_target(value, target_def)
        weight = float(target_def.get("weight", 0))

        met = True
        if "min" in target_def and value < target_def["min"]:
            met = False
        if "max" in target_def and value > target_def["max"]:
            met = False

        details[target_name] = {
            "value": value,
            "target": target_def.get("min") or target_def.get("max"),
            "score": round(score, 1),
            "met": met,
        }
        weighted_sum += score * weight
        total_weight += weight

    total_score = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0

    # Check constraints
    violations: List[str] = []
    trades = result.get("trades", 0)
    if constraints.get("min_trades") and trades < constraints["min_trades"]:
        violations.append(f"trades={trades} < min_trades={constraints['min_trades']}")

    # Generate suggestions
    suggestions = _generate_suggestions(details, violations)

    return {
        "total_score": total_score,
        "grade": _grade(total_score),
        "details": details,
        "constraint_violations": violations,
        "suggestions": suggestions,
    }


def _generate_suggestions(details: Dict[str, Any], violations: List[str]) -> List[str]:
    """Generate actionable improvement suggestions."""
    suggestions: List[str] = []
    if violations:
        suggestions.append(f"Fix constraint violations: {'; '.join(violations)}")

    # Find worst-scoring targets
    scored = [(name, d) for name, d in details.items() if d.get("value") is not None]
    scored.sort(key=lambda x: x[1]["score"])

    for name, d in scored[:3]:
        if d["met"]:
            continue
        val = d["value"]
        target = d["target"]
        if name == "sharpe":
            suggestions.append(f"Sharpe={val:.2f} (target≥{target}): improve signal quality or reduce trade frequency")
        elif name == "max_drawdown_pct":
            suggestions.append(f"DD={val:.1f}% (target≤{target}%): tighten stoploss or reduce position size")
        elif name == "profit_pct":
            suggestions.append(f"Profit={val:.2f}% (target≥{target}%): improve entry timing or hold winners longer")
        elif name == "win_rate":
            suggestions.append(f"WinRate={val:.1%} (target≥{target}): filter entries more strictly")
        elif name == "profit_factor":
            suggestions.append(f"PF={val:.2f} (target≥{target}): cut losers faster or let winners run")
        else:
            suggestions.append(f"{name}={val} needs improvement (target={target})")

    if not suggestions:
        suggestions.append("All targets met! Consider tightening objectives for further improvement.")

    return suggestions


__all__ = ["evaluate"]

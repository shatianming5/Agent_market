"""Factor-level scoring utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def compute_factor_score(
    features_parquet: Optional[Path] = None,
    expression: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Run factor-level scoring using factor_compiler.scoring if available.

    Returns scoring dict or None if dependencies are missing.
    """
    if features_parquet is None or expression is None:
        return None

    try:
        import pandas as pd
        from agent_market.factor_compiler.scoring.aggregate import score_factors_to_artifacts
    except ImportError:
        logger.debug("factor_compiler.scoring not available, skipping factor-level scoring")
        return None

    features_path = Path(features_parquet)
    if not features_path.exists():
        logger.debug("Features parquet not found: %s", features_path)
        return None

    try:
        df = pd.read_parquet(features_path)
    except Exception as e:
        logger.warning("Failed to read features parquet: %s", e)
        return None

    # Check for target column
    target_col = None
    for candidate in ("target", "return_1h", "return_4h", "close_pct"):
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        logger.debug("No target column found in features, skipping factor scoring")
        return None

    # Find factor columns (non-target numeric)
    factor_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c != target_col and not c.startswith("_")
    ]
    if not factor_cols:
        return None

    if out_dir is None:
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="factor_score_"))
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = score_factors_to_artifacts(
            df,
            factor_cols=factor_cols[:20],
            target_col=target_col,
            out_dir=out_dir,
        )
        items = result.get("items", [])
        if items:
            best = max(items, key=lambda x: abs(x.get("ic", 0) or 0))
            return {
                "best_factor": best.get("name"),
                "best_ic": best.get("ic"),
                "best_sharpe": best.get("sharpe_net"),
                "factors_scored": len(items),
                "pareto_front": result.get("pareto_front", []),
            }
    except Exception as e:
        logger.warning("Factor scoring failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# D6: Trace grading — evaluate agent reasoning quality
# ---------------------------------------------------------------------------

def grade_trace(candidate: Any) -> Dict[str, Any]:
    """Grade the quality of agent reasoning for a candidate.

    Produces a bounded score from available traces and validation results.
    Does NOT use LLM — purely rule-based for speed and determinism.
    """
    planner_notes = str(getattr(candidate, "planner_notes", "") or "")
    reviewer_notes = str(getattr(candidate, "reviewer_notes", "") or "")
    code = str(getattr(candidate, "code", "") or "")
    diagnosis = str(getattr(candidate, "diagnosis", "") or "")
    validation_passed = bool(getattr(candidate, "validation_passed", False))
    constraints_ok = bool(getattr(candidate, "constraints_ok", True))
    failure_category = str(getattr(candidate, "failure_category", "") or "")

    scores: Dict[str, float] = {}
    flags: list = []

    # 1) Did the planner produce a clear hypothesis?
    hypothesis_clear = len(planner_notes) > 30
    scores["hypothesis_clear"] = 1.0 if hypothesis_clear else 0.0

    # 2) Did the reviewer identify issues?
    reviewer_active = len(reviewer_notes) > 20
    scores["reviewer_active"] = 1.0 if reviewer_active else 0.0

    # 3) Was validation passed on first try?
    scores["validation_first_pass"] = 1.0 if validation_passed else 0.0

    # 4) Are constraints met?
    scores["constraints_met"] = 1.0 if constraints_ok else 0.0

    # 5) Was code non-trivial (> 50 lines)?
    code_lines = len(code.strip().splitlines()) if code else 0
    scores["code_nontrivial"] = 1.0 if code_lines > 50 else 0.5 if code_lines > 20 else 0.0

    # 6) Failure categorization
    if failure_category:
        flags.append(f"failed:{failure_category}")
        scores["no_failure"] = 0.0
    else:
        scores["no_failure"] = 1.0

    # 7) Wasted cycles indicator
    wasted = 0
    if "syntax" in diagnosis.lower():
        wasted += 1
        flags.append("syntax_error")
    if "import" in diagnosis.lower():
        wasted += 1
        flags.append("import_error")
    scores["no_wasted_cycles"] = max(0.0, 1.0 - wasted * 0.5)

    # Overall grade: weighted average
    weights = {
        "hypothesis_clear": 0.15,
        "reviewer_active": 0.10,
        "validation_first_pass": 0.20,
        "constraints_met": 0.20,
        "code_nontrivial": 0.10,
        "no_failure": 0.15,
        "no_wasted_cycles": 0.10,
    }
    overall = sum(scores.get(k, 0) * w for k, w in weights.items())

    return {
        "overall_grade": round(overall, 3),
        "scores": scores,
        "flags": flags,
        "code_lines": code_lines,
    }

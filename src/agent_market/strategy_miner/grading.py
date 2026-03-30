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

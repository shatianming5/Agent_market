from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_factor_scoring_writes_artifacts(tmp_path: Path):
    from agent_market.factor_compiler.scoring import score_factors_to_artifacts

    df = pd.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "factor_good": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "factor_nan": [None, None, 0.0, None, 0.0, None],
        }
    )

    out = score_factors_to_artifacts(
        df,
        factor_cols=["factor_good", "factor_nan"],
        target_col="y",
        out_dir=tmp_path,
        max_nan_ratio=0.2,
        max_turnover=10.0,
    )

    scores_path = Path(out["artifacts"]["factor_scores_json"])
    pareto_path = Path(out["artifacts"]["pareto_csv"])
    assert scores_path.exists()
    assert pareto_path.exists()

    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    assert payload["target_col"] == "y"
    items = {row["name"]: row for row in payload["items"]}
    assert items["factor_good"]["gate_pass"] is True
    assert items["factor_nan"]["gate_pass"] is False
    assert payload["pareto_front"]


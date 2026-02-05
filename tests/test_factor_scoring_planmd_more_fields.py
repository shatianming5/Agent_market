from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_factor_scoring_includes_additional_planmd_fields(tmp_path: Path) -> None:
    from agent_market.factor_compiler.scoring import score_factors_to_artifacts

    rng = np.random.default_rng(7)
    n = 400
    y = rng.normal(loc=0.0, scale=0.01, size=n)
    f = y + rng.normal(loc=0.0, scale=0.01, size=n)

    df = pd.DataFrame({"y": y, "factor": f})

    out = score_factors_to_artifacts(
        df,
        factor_cols=["factor"],
        target_col="y",
        out_dir=tmp_path,
        factor_exprs={"factor": "roll_mean(close,10) + roll_std(close,5)"},
    )

    payload = json.loads(Path(out["artifacts"]["factor_scores_json"]).read_text(encoding="utf-8"))
    row = payload["items"][0]

    for key in [
        "regime_consistency",
        "train_test_gap",
        "capacity_proxy",
        "expensive_ops",
        "slippage_reduction_bps",
        "fill_rate",
        "adverse_selection_proxy",
    ]:
        assert key in row

    assert row["expensive_ops"] is not None
    assert row["regime_consistency"] is None or 0.0 <= float(row["regime_consistency"]) <= 1.0
    assert row["train_test_gap"] is None or float(row["train_test_gap"]) >= 0.0
    assert row["capacity_proxy"] is None or float(row["capacity_proxy"]) >= 0.0


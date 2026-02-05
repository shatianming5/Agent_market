from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_factor_scoring_includes_planmd_fields(tmp_path: Path) -> None:
    from agent_market.factor_compiler.scoring import score_factors_to_artifacts

    rng = np.random.default_rng(7)
    n = 200
    y = rng.normal(loc=0.0, scale=0.01, size=n)
    f1 = y + rng.normal(loc=0.0, scale=0.01, size=n)
    f2 = rng.normal(loc=0.0, scale=1.0, size=n)

    df = pd.DataFrame({"y": y, "f1": f1, "f2": f2})

    out = score_factors_to_artifacts(
        df,
        factor_cols=["f1", "f2"],
        target_col="y",
        out_dir=tmp_path,
        factor_exprs={"f1": "ts_z(close,4)", "f2": "roll_mean(close,5)"},
        factor_library={"lib_factor": df["f1"]},
        library_expr_sha256=[],
        max_corr_to_library=0.95,
        ic_rolling_window=50,
    )

    scores_path = Path(out["artifacts"]["factor_scores_json"])
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    items = {row["name"]: row for row in payload["items"]}

    required = [
        "ic",
        "rank_ic",
        "ic_ir",
        "ic_rolling_std",
        "sharpe_net",
        "sortino_net",
        "mdd",
        "corr_to_library_max",
        "ast_similarity_max",
        "node_count",
        "depth",
        "weighted_score",
        "gate_pass",
    ]
    for name in ["f1", "f2"]:
        for key in required:
            assert key in items[name], (name, key)

    assert items["f1"]["corr_to_library_max"] is not None
    assert float(items["f1"]["corr_to_library_max"]) >= 0.99
    assert items["f1"]["gate_pass"] is False

    assert items["f2"]["gate_pass"] is True


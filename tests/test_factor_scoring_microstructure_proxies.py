from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_factor_scoring_computes_microstructure_proxy_fields_when_available(tmp_path: Path) -> None:
    from agent_market.factor_compiler.scoring import score_factors_to_artifacts

    rng = np.random.default_rng(7)
    n = 300

    # Microstructure proxy columns (shape roughly like the real outputs)
    expected_slippage_proxy = np.linspace(2.0, 12.0, n) + rng.normal(scale=0.2, size=n)
    expected_slippage_proxy = np.clip(expected_slippage_proxy, 0.0, None)
    fill_prob_proxy = 1.0 / (1.0 + np.exp((expected_slippage_proxy - 7.0)))
    toxicity_proxy = np.abs(rng.normal(loc=0.0, scale=1.0, size=n))

    # Make the factor "prefer" low slippage by having larger magnitude when slippage is low.
    factor = 1.0 / (expected_slippage_proxy + 1e-6) + rng.normal(scale=0.01, size=n)
    y = rng.normal(loc=0.0, scale=0.01, size=n)

    df = pd.DataFrame(
        {
            "y": y,
            "factor": factor,
            "expected_slippage_proxy": expected_slippage_proxy,
            "fill_prob_proxy": fill_prob_proxy,
            "toxicity_proxy": toxicity_proxy,
        }
    )

    out = score_factors_to_artifacts(
        df,
        factor_cols=["factor"],
        target_col="y",
        out_dir=tmp_path,
    )

    payload = json.loads(Path(out["artifacts"]["factor_scores_json"]).read_text(encoding="utf-8"))
    row = payload["items"][0]

    assert row["slippage_reduction_bps"] is not None
    assert row["fill_rate"] is not None
    assert row["adverse_selection_proxy"] is not None

    assert float(row["fill_rate"]) >= 0.0
    assert float(row["fill_rate"]) <= 1.0

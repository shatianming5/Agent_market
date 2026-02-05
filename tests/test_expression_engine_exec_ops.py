from __future__ import annotations

import numpy as np
import pandas as pd


def test_expression_engine_exec_ops_on_microstructure_like_df() -> None:
    from agent_market.freqai.expression_engine import safe_eval_expression

    df = pd.DataFrame(
        {
            "mid": [100.0, 101.0, 99.5, 100.5, 100.2],
            "spread": [0.1, 0.12, 0.09, 0.11, 0.1],
            "depth_bid_20": [10.0, 11.0, 9.0, 10.5, 10.2],
            "depth_ask_20": [9.0, 9.5, 10.0, 9.8, 9.9],
            "imbalance_20": [0.05, 0.10, -0.05, 0.0, 0.02],
            "arrival_intensity_10": [1.0, 0.5, 1.5, 2.0, 1.2],
            "ofi_10": [0.1, -0.2, 0.05, 0.0, 0.3],
        }
    )

    prob = safe_eval_expression("fill_prob(0,10)", df)
    assert prob.shape[0] == df.shape[0]
    assert float(prob.min()) >= 0.0
    assert float(prob.max()) <= 1.0

    impact = safe_eval_expression("impact_proxy(10)", df)
    assert impact.shape[0] == df.shape[0]
    assert np.isfinite(float(impact.fillna(0.0).iloc[-1]))

    q = safe_eval_expression("queue_pos_proxy()", df)
    assert q.shape[0] == df.shape[0]
    assert float(q.min()) >= 0.0
    assert float(q.max()) <= 1.0


def test_expression_engine_exec_ops_fallbacks_on_close_only_df() -> None:
    from agent_market.freqai.expression_engine import safe_eval_expression

    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 101.5]})

    prob = safe_eval_expression("fill_prob(0,10)", df)
    assert prob.shape[0] == df.shape[0]
    assert np.allclose(prob.to_numpy(), 0.5, atol=1e-12)

    impact = safe_eval_expression("impact_proxy(10)", df)
    assert impact.shape[0] == df.shape[0]
    assert float(impact.fillna(0.0).max()) == 0.0

    q = safe_eval_expression("queue_pos_proxy()", df)
    assert q.shape[0] == df.shape[0]
    assert np.allclose(q.to_numpy(), 0.5, atol=1e-12)


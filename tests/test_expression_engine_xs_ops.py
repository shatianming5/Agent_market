from __future__ import annotations

import numpy as np
import pandas as pd


def test_expression_engine_xs_ops_rank_zscore_corr_neutralize() -> None:
    from agent_market.freqai.expression_engine import safe_eval_expression

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")] * 3 + [pd.Timestamp("2026-01-02")] * 3,
            "pair": ["A", "B", "C", "A", "B", "C"],
            "x": [1.0, 2.0, 3.0, 10.0, 0.0, -10.0],
            "y": [2.0, 4.0, 6.0, 1.0, -1.0, -3.0],
            "a": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        }
    )

    r1 = safe_eval_expression("rank_xs(x)", df)
    r2 = safe_eval_expression("rank_xs(x,date)", df)
    assert np.allclose(r1.to_numpy(), r2.to_numpy(), equal_nan=True)
    assert r1.iloc[0] < r1.iloc[1] < r1.iloc[2]
    assert r1.iloc[3] > r1.iloc[4] > r1.iloc[5]

    z = safe_eval_expression("zscore_xs(x)", df)
    for d, g in df.groupby("date", sort=False):
        sg = z.loc[g.index]
        assert abs(float(sg.mean())) < 1e-6, d
        assert abs(float(sg.std(ddof=0)) - 1.0) < 1e-6, d

    c = safe_eval_expression("corr_xs(x,y)", df)
    g1 = c.loc[df.index[:3]]
    assert np.allclose(g1.to_numpy(), np.ones(3), atol=1e-9, equal_nan=False)

    x_lin = pd.Series([5.0, 7.0, 9.0, 1.0, 4.0, 7.0], index=df.index)
    df2 = df.copy()
    df2["x_lin"] = x_lin
    resid = safe_eval_expression("neutralize(x_lin,a)", df2)
    assert float(resid.abs().max()) < 1e-9

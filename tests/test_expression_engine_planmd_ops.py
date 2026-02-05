from __future__ import annotations

import numpy as np
import pandas as pd


def test_expression_engine_planmd_ops() -> None:
    from agent_market.freqai.expression_engine import safe_eval_expression

    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 100.0, 5.0]})

    out = safe_eval_expression("diff(close,1)", df)
    assert np.isfinite(out.dropna()).all()
    assert float(out.iloc[1]) == 1.0

    out2 = safe_eval_expression("decay_linear(close,3)", df)
    assert len(out2) == len(df)
    assert np.isnan(out2.iloc[0]) and np.isnan(out2.iloc[1])
    assert np.isfinite(out2.iloc[-1])

    out3 = safe_eval_expression("winsorize(close,0.2)", df)
    assert len(out3) == len(df)
    assert float(out3.max()) <= float(df["close"].quantile(0.8)) + 1e-9

    out4 = safe_eval_expression("robust_z(close,3)", df)
    assert len(out4) == len(df)


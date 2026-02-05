from __future__ import annotations

import numpy as np
import pandas as pd


def test_check_shift_test_flags_high_autocorr_case() -> None:
    from agent_market.factor_compiler.checks import check_shift_test

    rng = np.random.default_rng(7)
    y = pd.Series(np.cumsum(rng.normal(loc=0.0, scale=1.0, size=300)))
    x = y.copy()

    res = check_shift_test(x, y, shift=1, corr_threshold=0.98)
    assert res.ok is False
    assert res.code == "LEAKAGE_SHIFT_SUSPECT"


def test_check_label_leakage_signature_flags_0_lag_spike() -> None:
    from agent_market.factor_compiler.checks import check_label_leakage_signature

    rng = np.random.default_rng(7)
    y = pd.Series(rng.normal(loc=0.0, scale=1.0, size=300))
    x = y + pd.Series(rng.normal(loc=0.0, scale=1e-3, size=300))

    res = check_label_leakage_signature(x, y, max_lag=3, corr_threshold=0.98, spike_ratio=0.5)
    assert res.ok is False
    assert res.code == "LEAKAGE_SIGNATURE_SPIKE"


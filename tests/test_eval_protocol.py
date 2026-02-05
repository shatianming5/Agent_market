from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def test_walkforward_splits_respect_purge_and_embargo():
    from agent_market.freqai.training.eval_protocol import walkforward_splits

    splits = walkforward_splits(50, train_size=10, test_size=5, step_size=5, purge=2, embargo=3)
    assert splits
    for sp in splits:
        assert sp.test_start >= sp.train_end + sp.purge


def test_cost_accounting_net_profit_abs():
    from agent_market.freqai.training.eval_protocol import compute_cost_adjusted_metrics

    backtest = {"profit_total_abs": 10.0}
    tca = {"summary": {"fees_total": 1.5}}
    out = compute_cost_adjusted_metrics(backtest, tca)
    assert out["net_profit_abs"] == pytest.approx(8.5)


def test_eval_protocol_time_safety_rejects_negative_shift():
    from agent_market.freqai.expression_engine import ExpressionValidationError, safe_eval_expression

    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ExpressionValidationError):
        safe_eval_expression("shift(close,-1)", df)


def test_eval_protocol_script_writes_report(tmp_path: Path):
    from agent_market.freqai.training.eval_protocol import build_eval_report, write_eval_report

    run_meta = {
        "run_id": "abc123",
        "config": {"sha256": "deadbeef"},
        "artifacts": {"feedback_summary": "bt.json", "tca_report": "tca.json"},
        "paths": {"run_meta": "artifacts/runs/abc123/run_meta.json"},
    }
    backtest = {"profit_total_abs": 10.0}
    tca = {"summary": {"fees_total": 2.0}}

    payload = build_eval_report(run_meta=run_meta, backtest_summary=backtest, tca_report=tca)
    out_path = write_eval_report(tmp_path / "eval_report.json", payload)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["net_profit_abs"] == pytest.approx(8.0)


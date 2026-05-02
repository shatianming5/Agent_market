from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_market.factor_lab import mine_lean_gate, mining


def test_loop_lean_gate_uses_current_state_and_records_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mining, "LAB_STATE", tmp_path / "factor_lab")
    captured: dict = {}

    def fake_run_mine_lean_gate(**kwargs):
        captured.update(kwargs)
        summary = Path(kwargs["output"]) / "mine_lean_gate.json"
        comparison = Path(kwargs["output"]) / "comparison.json"
        project = Path(kwargs["output"]) / "lean_project"
        summary.parent.mkdir(parents=True, exist_ok=True)
        return {
            "status": mine_lean_gate.STATUS_PASSED,
            "reason": "unit passed",
            "comparison_status": "ok",
            "duration_sec": 1.25,
            "artifacts": {
                "summary": str(summary),
                "comparison_json": str(comparison),
                "lean_project": str(project),
                "lean_result": str(project / "results.json"),
            },
        }

    monkeypatch.setattr(mine_lean_gate, "run_mine_lean_gate", fake_run_mine_lean_gate)
    cfg = mining.MiningConfig(
        rounds=5,
        timeframe="4h",
        data_venue="binance",
        lean_gate_every=1,
        lean_gate_n=7,
        lean_gate_venue="auto",
        lean_gate_data_venue="auto",
        lean_gate_bin="lean-unit",
        lean_gate_timeout=123,
        lean_gate_min_final_equity=0.0,
        lean_gate_min_trades=0,
        lean_gate_rank_top_k=3,
        lean_gate_score_threshold=0.75,
        lean_gate_recompute_corr=True,
    )
    survivors = [mining.CandidateRecord(expression="close", origin="unit", oos_ic=0.05)]
    mining.save_state("unit_hook", 3, survivors, {"close"}, cfg)

    result = mining._run_loop_lean_gate("unit_hook", 3, cfg)

    state_path = tmp_path / "factor_lab" / "mining" / "unit_hook" / "state_0003.json"
    assert result["status"] == mine_lean_gate.STATUS_PASSED
    assert captured["candidate_state"] == state_path
    assert captured["run_id"] == "loop_0003"
    assert captured["venue"] == "binance"
    assert captured["data_venue"] == "binance"
    assert captured["timeframe"] == "4h"
    assert captured["lean_bin"] == "lean-unit"
    assert captured["lean_timeout"] == 123
    assert captured["min_final_equity"] == 0.0
    assert captured["min_trades"] == 0
    assert captured["rank_kwargs"]["top_k"] == 3
    assert captured["rank_kwargs"]["min_abs_score_z"] == 0.75
    assert captured["rank_kwargs"]["recompute_corr"] is True

    history_path = tmp_path / "factor_lab" / "mining" / "unit_hook" / "lean_gate_history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert history[-1]["loop"] == 3
    assert history[-1]["status"] == mine_lean_gate.STATUS_PASSED
    latest = json.loads((tmp_path / "factor_lab" / "mining" / "unit_hook" / "latest.json").read_text(encoding="utf-8"))
    assert latest["lean_gate_latest"]["comparison_status"] == "ok"


def test_loop_lean_gate_schedule_runs_interval_and_final() -> None:
    cfg = mining.MiningConfig(rounds=11, lean_gate_every=5)

    assert not mining._should_run_loop_lean_gate(cfg, 1)
    assert mining._should_run_loop_lean_gate(cfg, 5)
    assert mining._should_run_loop_lean_gate(cfg, 10)
    assert mining._should_run_loop_lean_gate(cfg, 11)


def test_llm_required_stops_when_no_usable_llm_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mining, "LAB_STATE", tmp_path / "factor_lab")
    monkeypatch.setattr(mining, "_open_hub_client", lambda tag: None)
    dates = pd.date_range("2025-01-01", periods=400, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "__pair__": ["BTC/USDT"] * len(dates),
            "__fwd_ret__": np.linspace(-0.01, 0.01, len(dates)),
            "close": np.linspace(100.0, 120.0, len(dates)),
        }
    )
    monkeypatch.setattr(mining, "build_big", lambda **kwargs: (frame, ["close"]))
    monkeypatch.setattr(mining, "load_seeds", lambda: ["close"])
    monkeypatch.setattr(
        mining,
        "eval_ic",
        lambda *args, **kwargs: {
            "status": "ok",
            "passes": True,
            "train_ic": 0.05,
            "oos_ic": 0.05,
            "sign_agree": 1,
            "combined": 0.05,
            "fitness": 0.05,
        },
    )
    monkeypatch.setattr(mining, "_llm_generate", lambda *args, **kwargs: [])

    cfg = mining.MiningConfig(
        rounds=1,
        top_k=1,
        use_llm=True,
        llm_required=True,
        py_per_loop=0,
        ic_gate=0.0,
        sign_gate=0,
        novelty_gate=1.0,
        hard_corr_gate=1.0,
        no_cache=True,
    )

    with pytest.raises(RuntimeError, match="LLM required"):
        mining.mine(cfg, tag="unit_llm_required", resume=False)

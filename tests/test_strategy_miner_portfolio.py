from __future__ import annotations

from pathlib import Path

from agent_market.strategy_miner._portfolio import build_candidate_portfolio_report
from agent_market.strategy_miner.dtypes import StrategyCandidate


def _candidate(name: str, reward: float, daily_profit: list[list[object]]) -> StrategyCandidate:
    candidate = StrategyCandidate(
        name=name,
        code="class X: pass\n",
        strategy_path=Path(f"/tmp/{name}.py"),
        iteration=1,
    )
    candidate.reward = reward
    candidate.constraints_ok = True
    candidate.backtest_summary = {
        "profit_total_pct": reward * 10.0,
        "max_drawdown_pct": 5.0,
        "daily_profit": daily_profit,
        "starting_balance": 1000.0,
    }
    return candidate


def test_build_candidate_portfolio_report_prunes_high_correlation(monkeypatch) -> None:
    import agent_market.portfolio_opt as portfolio_opt

    def _fake_optimize_hrp(returns_df):  # noqa: ANN001
        return {
            "weights": {column: (0.8 if idx == 0 else 0.2) for idx, column in enumerate(returns_df.columns)},
            "stats": {"n_assets": returns_df.shape[1]},
        }

    monkeypatch.setattr(portfolio_opt, "optimize_hrp", _fake_optimize_hrp)

    candidate_a = _candidate(
        "A",
        1.1,
        [
            ["2026-03-12", 12.0],
            ["2026-03-13", -3.0],
            ["2026-03-14", 9.0],
            ["2026-03-15", -2.0],
            ["2026-03-16", 8.0],
        ],
    )
    candidate_b = _candidate(
        "B",
        1.0,
        [
            ["2026-03-12", -8.0],
            ["2026-03-13", 6.0],
            ["2026-03-14", -5.0],
            ["2026-03-15", 7.0],
            ["2026-03-16", -4.0],
        ],
    )
    candidate_c = _candidate(
        "C",
        0.95,
        [
            ["2026-03-12", 11.5],
            ["2026-03-13", -2.8],
            ["2026-03-14", 8.7],
            ["2026-03-15", -1.9],
            ["2026-03-16", 7.8],
        ],
    )

    report = build_candidate_portfolio_report(
        [candidate_a, candidate_b, candidate_c],
        top_k=3,
        min_candidates=2,
        correlation_threshold=0.85,
        max_weight=0.60,
    )

    assert report is not None
    assert report["method"] == "hrp"
    assert set(report["weights"]) == {"A", "B"}
    assert report["weights"]["A"] <= 0.60
    assert any(item["name"] == "C" and item["reason"] == "correlation_pruned" for item in report["dropped_candidates"])

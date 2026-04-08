from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_choose_top_expression_prefers_highest_score(tmp_path: Path) -> None:
    from ws_production.factor_cycle import _choose_top_expression

    expr_path = tmp_path / "freqai_expressions_selected.json"
    expr_path.write_text(
        json.dumps(
            {
                "exchange": "gate",
                "timeframe": "1h",
                "pairs": ["BTC/USDT"],
                "expressions": [
                    {"name": "f001", "expression": "close", "score": 0.11, "origin": "classic"},
                    {"name": "f002", "expression": "ema(close, 12)", "score": 0.27, "origin": "llm"},
                    {"name": "f003", "expression": "z(close)", "score": 0.19, "origin": "classic"},
                ],
            }
        ),
        encoding="utf-8",
    )

    top = _choose_top_expression(expr_path)

    assert top["name"] == "f002"
    assert top["expression"] == "ema(close, 12)"
    assert top["score"] == 0.27
    assert top["pairs"] == ["BTC/USDT"]


def test_run_factor_cycle_writes_summary_and_invokes_expected_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from ws_production import factor_cycle as module

    workspace = tmp_path / "ws_production"
    (workspace / "results").mkdir(parents=True, exist_ok=True)

    freqai_config = tmp_path / "config_freqai_gate_1h.json"
    freqai_config.write_text(
        json.dumps(
            {
                "timeframe": "1h",
                "exchange": {"name": "gate", "pair_whitelist": ["BTC/USDT"]},
                "datadir": "user_data/data",
            }
        ),
        encoding="utf-8",
    )
    feature_file = tmp_path / "freqai_features_real.json"
    expressions_file = tmp_path / "freqai_expressions_selected.json"
    calls: list[list[str]] = []

    def _fake_run_command(cmd: list[str], *, cwd: Path = module.ROOT) -> dict[str, object]:
        calls.append([str(item) for item in cmd])
        script_name = Path(str(cmd[1])).name
        if script_name == "freqai_feature_agent.py":
            feature_file.write_text(json.dumps({"features": [], "feature_combos": []}), encoding="utf-8")
        elif script_name == "freqai_expression_agent.py":
            expressions_file.write_text(
                json.dumps(
                    {
                        "exchange": "gate",
                        "timeframe": "1h",
                        "pairs": ["BTC/USDT"],
                        "expressions": [
                            {"name": "f001", "expression": "close", "score": 0.11, "origin": "classic"},
                            {"name": "f002", "expression": "ema(close, 12)", "score": 0.27, "origin": "llm"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
        elif script_name == "verify_factors.py":
            report_dir = workspace / "results" / "factor_validation"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "factor_validation_20260408-170000.json").write_text("{}", encoding="utf-8")
        elif script_name == "factor_eval.py":
            out_dir = Path(str(cmd[cmd.index("--out-dir") + 1]))
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "factor_eval_meta.json").write_text("{}", encoding="utf-8")
            (out_dir / "factor_scores.json").write_text("{}", encoding="utf-8")
            (out_dir / "pareto.csv").write_text("factor,score\nf002,0.27\n", encoding="utf-8")
        return {
            "cmd": [str(item) for item in cmd],
            "cwd": str(cwd),
            "returncode": 0,
            "elapsed_sec": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    def _fake_build_dataset(*, freqai_config: Path, feature_file: Path, out_path: Path, pair: str | None = None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("stub", encoding="utf-8")
        return {
            "pair": pair or "BTC/USDT",
            "exchange": "gate",
            "timeframe": "1h",
            "rows": 128,
            "features_parquet": str(out_path),
        }

    monkeypatch.setattr(module, "_run_command", _fake_run_command)
    monkeypatch.setattr(module, "_build_factor_eval_dataset", _fake_build_dataset)

    summary = module.run_factor_cycle(
        workspace=workspace,
        model="custom/gpt-5.2",
        freqai_config=freqai_config,
        feature_file=feature_file,
        expressions_file=expressions_file,
        llm_enabled=True,
        top_n=12,
        llm_count=9,
        llm_loops=2,
    )

    assert summary["ok"] is True
    assert summary["top_expression"]["name"] == "f002"
    assert Path(summary["validation_report"]).exists()
    assert Path(summary["factor_eval_artifacts"]["factor_scores_json"]).exists()
    latest = json.loads((workspace / "results" / "factor_cycle_latest.json").read_text(encoding="utf-8"))
    assert latest["top_expression"]["name"] == "f002"
    assert [Path(cmd[1]).name for cmd in calls] == [
        "freqai_feature_agent.py",
        "freqai_expression_agent.py",
        "verify_factors.py",
        "factor_eval.py",
    ]
    expression_cmd = next(cmd for cmd in calls if Path(cmd[1]).name == "freqai_expression_agent.py")
    assert "--llm-enabled" in expression_cmd
    assert "--llm-provider" in expression_cmd
    assert "opencode" in expression_cmd
    assert "--llm-model" in expression_cmd
    assert "custom/gpt-5.2" in expression_cmd


def test_run_agent_loop_wires_factor_cycle() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "ws_production" / "run_agent_loop.sh").read_text(encoding="utf-8")

    assert "ws_production_factor_cycle.py" in script
    assert 'WS_FACTOR_LLM_ENABLED:-0' in script
    assert "--no-llm" in script
    assert "factor_cycle_latest.json" in script
    assert "Review the latest factor mining/evaluation outputs" in script


def test_factor_cycle_cli_wrapper_imports_from_repo_root() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/ws_production_factor_cycle.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Run one ws_production factor mining + evaluation cycle." in proc.stdout

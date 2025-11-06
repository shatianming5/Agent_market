import json
import re
import itertools
from pathlib import Path

import pytest

import scripts.agent_iterate as agent_iterate


@pytest.mark.parametrize("top_expressions", [1])
def test_agent_iterate_generates_report(tmp_path, monkeypatch, top_expressions):
    # Prepare dummy files
    config_path = tmp_path / "config.json"
    feature_path = tmp_path / "features.json"
    expression_path = tmp_path / "expressions.json"
    compute_output = tmp_path / "factor_ic.json"
    corr_output = tmp_path / "corr.csv"
    report_file = tmp_path / "iterations.json"
    backtest_prefix = tmp_path / "backtests"

    config_path.write_text("{}", encoding="utf-8")
    feature_path.write_text("{}", encoding="utf-8")
    expression_path.write_text(
        json.dumps(
            {
                "expressions": [
                    {"name": "expr_high", "enabled": True},
                    {"name": "expr_low", "enabled": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    # Mock run_command to avoid invoking external processes
    meta_counter = itertools.count()
    backtest_root = backtest_prefix.parent

    def fake_run_command(cmd, env=None, cwd=None):
        if isinstance(cmd, list):
            if "scripts/compute_factor_ic.py" in cmd:
                output_idx = cmd.index("--output") + 1
                output_path = Path(cmd[output_idx])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                summary = {
                    "features": [],
                    "expressions": [
                        {
                            "name": "expr_high",
                            "ic_pearson": 0.2,
                            "ic_spearman": 0.1,
                            "samples": 100,
                        },
                        {
                            "name": "expr_low",
                            "ic_pearson": 0.05,
                            "ic_spearman": 0.02,
                            "samples": 80,
                        },
                    ],
                }
                output_path.write_text(json.dumps(summary), encoding="utf-8")
                # Simulate corr output if requested
                if "--corr-output" in cmd:
                    corr_path = Path(cmd[cmd.index("--corr-output") + 1])
                    corr_path.parent.mkdir(parents=True, exist_ok=True)
                    corr_path.write_text("expr_high,expr_low\n", encoding="utf-8")
                return ""
            raise AssertionError(f"Unexpected command list: {cmd}")

        # Handle backtest command string
        match = re.search(r"--export-filename\s+(\S+)", cmd)
        assert match, f"Expected export filename in command: {cmd}"
        export_path = Path(match.group(1))
        export_path.parent.mkdir(parents=True, exist_ok=True)

        backtest_root.mkdir(parents=True, exist_ok=True)
        idx = next(meta_counter)
        meta_path = backtest_root / f"backtest-result-test-{idx:02d}.meta.json"
        zip_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".zip"))

        meta_payload = {
            "strategy": {
                "ExpressionLongStrategy": {
                    "profit_total_abs": 12.34,
                    "trades": [],
                }
            }
        }
        meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
        zip_path.write_bytes(b"")  # placeholder archive
        (backtest_root / ".last_result.json").write_text(
            json.dumps({"latest_backtest": zip_path.name}), encoding="utf-8"
        )
        return "ok"

    monkeypatch.setattr(agent_iterate, "run_command", fake_run_command)

    args = [
        "agent_iterate.py",
        "--config",
        str(config_path),
        "--feature-file",
        str(feature_path),
        "--expression-file",
        str(expression_path),
        "--iterations",
        "1",
        "--top-expressions",
        str(top_expressions),
        "--report-file",
        str(report_file),
        "--compute-output",
        str(compute_output),
        "--corr-output",
        str(corr_output),
        "--backtest-command",
        "freqtrade backtesting --config {config} --export-filename {export}",
        "--backtest-export-prefix",
        str(backtest_prefix),
    ]

    monkeypatch.setenv("FREQAI_EXPRESSIONS_FILE", str(expression_path))
    monkeypatch.setenv("FREQAI_FEATURES_FILE", str(feature_path))

    monkeypatch.setattr("sys.argv", args)
    agent_iterate.main()

    # Load iteration report and validate structure
    report = json.loads(report_file.read_text(encoding="utf-8"))
    iterations = report.get("iterations", [])
    assert len(iterations) == 1
    iteration_entry = iterations[0]
    assert iteration_entry["selected_expressions"][0]["name"] == "expr_high"
    assert "backtest" in iteration_entry
    assert iteration_entry["backtest"]["meta_file"] is not None

    # Ensure expression enable flags updated
    updated_exprs = json.loads(expression_path.read_text(encoding="utf-8"))["expressions"]
    enabled_flags = {expr["name"]: expr.get("enabled") for expr in updated_exprs}
    assert enabled_flags["expr_high"] is True
    assert enabled_flags["expr_low"] is False

    # compute output should be cleaned up
    assert not compute_output.exists()

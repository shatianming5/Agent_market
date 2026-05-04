"""tried_log append/read/format tests."""
from __future__ import annotations

import json
from pathlib import Path

from agent_market.wq_brain.tried_log import append_tried, format_for_prompt, read_tried


def test_append_creates_jsonl(tmp_path: Path):
    p = tmp_path / "tried" / "tag1.jsonl"
    append_tried(p, expr="rank(close)", sharpe=1.5, fitness=1.2, turnover=0.18,
                 alpha_id="A1", status="COMPLETE")
    assert p.exists()
    line = p.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["expr"] == "rank(close)"
    assert rec["sharpe"] == 1.5
    assert rec["alpha_id"] == "A1"


def test_append_multiple_entries(tmp_path: Path):
    p = tmp_path / "tag1.jsonl"
    for i, expr in enumerate(["rank(close)", "rank(open)", "rank(volume)"]):
        append_tried(p, expr=expr, sharpe=float(i), fitness=float(i) * 0.5,
                     turnover=0.2, alpha_id=f"A{i}", status="COMPLETE")
    records = read_tried(p)
    assert len(records) == 3
    assert {r["expr"] for r in records} == {"rank(close)", "rank(open)", "rank(volume)"}


def test_read_tail_limits(tmp_path: Path):
    p = tmp_path / "tag1.jsonl"
    for i in range(20):
        append_tried(p, expr=f"rank(field_{i})", sharpe=1.0, fitness=1.0,
                     turnover=0.2, alpha_id=f"A{i}", status="COMPLETE")
    records = read_tried(p, tail=5)
    assert len(records) == 5
    # Tail should be the LAST 5 entries
    exprs = [r["expr"] for r in records]
    assert exprs[-1] == "rank(field_19)"


def test_read_missing_file_returns_empty(tmp_path: Path):
    assert read_tried(tmp_path / "nope.jsonl") == []


def test_read_tolerates_corrupt_line(tmp_path: Path):
    p = tmp_path / "tag.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '{"expr": "rank(close)", "sharpe": 1.5}\n'
        'not valid json\n'
        '{"expr": "rank(open)", "sharpe": 1.0}\n',
        encoding="utf-8",
    )
    records = read_tried(p)
    assert len(records) == 2  # corrupt line skipped


def test_format_for_prompt_renders_table(tmp_path: Path):
    records = [
        {"expr": "rank(close)", "sharpe": 1.5, "fitness": 1.2, "turnover": 0.18,
         "status": "COMPLETE", "ts": 100.0},
        {"expr": "rank(open)", "sharpe": -0.5, "fitness": -0.2, "turnover": 0.30,
         "status": "COMPLETE", "ts": 200.0},
        {"expr": "rank(volume)", "sharpe": None, "fitness": None, "turnover": None,
         "status": "ERROR", "error": "compile failure", "ts": 300.0},
    ]
    out = format_for_prompt(records)
    assert "rank(close)" in out
    assert "1.50" in out
    assert "rank(open)" in out
    assert "-0.50" in out
    assert "rank(volume)" in out
    assert "ERROR" in out
    # Passing alphas should appear before failures
    assert out.index("rank(close)") < out.index("rank(open)")


def test_format_for_prompt_dedupes_same_expr_keeping_latest(tmp_path: Path):
    records = [
        {"expr": "rank(close)", "sharpe": 0.5, "fitness": 0.3, "turnover": 0.2,
         "status": "COMPLETE", "ts": 100.0},
        {"expr": "rank(close)", "sharpe": 1.5, "fitness": 1.2, "turnover": 0.18,
         "status": "COMPLETE", "ts": 200.0},
    ]
    out = format_for_prompt(records)
    # Should only have one row for rank(close), with the LATER (1.5/1.2) values
    assert out.count("rank(close)") == 1
    assert "1.50" in out
    assert "0.50" not in out


def test_format_for_prompt_empty_returns_empty():
    assert format_for_prompt([]) == ""

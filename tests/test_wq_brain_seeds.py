"""Seed template + grid expansion tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.wq_brain.seeds import SeedTemplate, expand_all, load_seeds_config


def test_template_no_placeholders_returns_as_is():
    t = SeedTemplate("rank(close)")
    assert t.placeholders() == []
    assert t.expand({}) == ["rank(close)"]


def test_template_single_placeholder_expansion():
    t = SeedTemplate("rank(ts_mean(close, <WIN>))")
    assert t.placeholders() == ["<WIN>"]
    out = t.expand({"<WIN>": [5, 10, 20]})
    assert set(out) == {
        "rank(ts_mean(close, 5))",
        "rank(ts_mean(close, 10))",
        "rank(ts_mean(close, 20))",
    }


def test_template_multi_placeholder_cartesian_product():
    t = SeedTemplate("rank(ts_corr(<F1>, <F2>, <WIN>))")
    out = t.expand({
        "<F1>": ["close", "vwap"],
        "<F2>": ["volume"],
        "<WIN>": [5, 10],
    })
    assert len(out) == 4
    assert "rank(ts_corr(close, volume, 5))" in out
    assert "rank(ts_corr(vwap, volume, 10))" in out


def test_template_repeated_placeholder_substituted_consistently():
    t = SeedTemplate("rank((<F1> - ts_mean(<F1>, <WIN>)) / <F1>)")
    out = t.expand({"<F1>": ["close"], "<WIN>": [20]})
    assert out == ["rank((close - ts_mean(close, 20)) / close)"]


def test_template_missing_placeholder_in_grid_raises():
    t = SeedTemplate("rank(ts_mean(<F1>, <WIN>))")
    with pytest.raises(ValueError, match="not in grid"):
        t.expand({"<F1>": ["close"]})  # <WIN> missing


def test_load_seeds_config_round_trip(tmp_path: Path):
    cfg = {
        "seeds": [
            "rank(close)",
            "rank(ts_mean(<F1>, <WIN>))",
        ],
        "grid": {"<F1>": ["close"], "<WIN>": [5, 10]},
    }
    p = tmp_path / "seeds.json"
    p.write_text(json.dumps(cfg))
    seeds, grid = load_seeds_config(p)
    assert len(seeds) == 2
    assert seeds[0].pattern == "rank(close)"
    assert grid["<WIN>"] == [5, 10]


def test_expand_all_combines_templates(tmp_path: Path):
    seeds = [
        SeedTemplate("rank(close)"),
        SeedTemplate("rank(<F1>)"),
    ]
    grid = {"<F1>": ["close", "open"]}
    out = expand_all(seeds, grid)
    # 1 + 2 = 3
    assert len(out) == 3
    assert "rank(close)" in out
    assert "rank(open)" in out


def test_real_seeds_config_loads():
    project_root = Path(__file__).resolve().parents[1]
    p = project_root / "configs" / "wqb_seeds_v1.json"
    assert p.exists(), f"missing {p}"
    seeds, grid = load_seeds_config(p)
    assert len(seeds) >= 8
    assert "<WIN>" in grid
    out = expand_all(seeds, grid)
    assert len(out) >= 100

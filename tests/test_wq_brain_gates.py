"""Tests for the cool-down hard gate + lex conflict resolver."""
from __future__ import annotations

import importlib.util
import io
import json
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

from agent_market.wq_brain.gates import (
    CooldownBlock,
    lex_precedence_key,
    lex_resolve_conflicts,
    slot_cooldown_block,
)
from agent_market.wq_brain.pheromone_cache import PheromoneLink
from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "wq_brain.py"


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def _row(expr, *, ts, sharpe=1.3, fitness=1.0, turnover=0.2, delta_U=None, alpha_id=None):
    out = {
        "ts": ts,
        "expr": expr,
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        "alpha_id": alpha_id or f"a_{expr[:6]}",
        "status": "COMPLETE",
        "region": "USA",
        "universe": "TOP500",
        "decay": 6,
    }
    if delta_U is not None:
        out["delta_U"] = delta_U
    return out


def test_cooldown_block_fires_on_repeat_with_low_delta_u():
    now = time.time()
    rows = [
        _row("rank(close) * rank(volume)", ts=now - 60,
             delta_U=0.01, alpha_id="orig"),
    ]
    blk = slot_cooldown_block(
        "rank(close) * rank(volume)",  # same skeleton, same fields
        rows,
        window=8,
        delta_flip=0.05,
        now=now,
    )
    assert isinstance(blk, CooldownBlock)
    assert blk.last_seen_alpha_id == "orig"
    assert "cool-down" in blk.reason or "δ_flip" in blk.reason


def test_cooldown_block_lifts_when_prior_delta_u_above_threshold():
    now = time.time()
    rows = [
        _row("rank(close) * rank(volume)", ts=now - 60,
             delta_U=0.20, alpha_id="orig"),
    ]
    assert (
        slot_cooldown_block(
            "rank(close) * rank(volume)", rows, now=now,
        )
        is None
    )


def test_cooldown_block_returns_none_when_slot_unseen():
    now = time.time()
    rows = [_row("rank(volume)", ts=now - 60, delta_U=0.01)]
    assert (
        slot_cooldown_block("rank(close) * rank(returns)", rows, now=now)
        is None
    )


def test_cooldown_block_handles_legacy_rows_missing_delta_u():
    now = time.time()
    parent = _row("rank(volume)", ts=now - 120, sharpe=1.10, fitness=0.90, turnover=0.30)
    child = _row("rank(close) * rank(volume)", ts=now - 60,
                 sharpe=1.05, fitness=0.85, turnover=0.32)
    blk = slot_cooldown_block(
        "rank(close) * rank(volume)",
        [parent, child],
        now=now,
    )
    # ΔU computed from utility_score should be negative → block fires.
    assert isinstance(blk, CooldownBlock)


def _make_link(
    *,
    altitude=ALTITUDE_L2_OP_FAMILY,
    expr="rank(close) * rank(volume)",
    evidence_type="op_swap",
    delta_U=0.3,
    support=0,
    ts=None,
    alpha_id="lex_a",
):
    return PheromoneLink(
        altitude=altitude,
        expr=expr,
        alpha_id=alpha_id,
        region="USA",
        universe="TOP500",
        sharpe=1.5,
        fitness=1.1,
        turnover=0.20,
        delta_U=delta_U,
        evidence_type=evidence_type,
        parent_alpha_id=None,
        source_panel_tag="src",
        ts=ts if ts is not None else time.time(),
        support=support,
        conflicts=0,
    )


def test_lex_precedence_hard_evidence_beats_soft():
    soft = _make_link(evidence_type="op_swap", alpha_id="soft")
    hard = _make_link(evidence_type="region_swap", alpha_id="hard")
    assert lex_precedence_key(hard) > lex_precedence_key(soft)


def test_lex_precedence_higher_altitude_beats_lower():
    low = _make_link(altitude=ALTITUDE_L4_NUMERIC_TWEAK, alpha_id="l4")
    high = _make_link(altitude=ALTITUDE_L1_REGION_UNIVERSE, alpha_id="l1",
                      evidence_type="op_swap")  # both soft, only altitude differs
    assert lex_precedence_key(high) > lex_precedence_key(low)


def test_lex_precedence_higher_support_beats_lower_when_tied():
    a = _make_link(support=2, alpha_id="a")
    b = _make_link(support=5, alpha_id="b")
    assert lex_precedence_key(b) > lex_precedence_key(a)


def test_lex_resolve_conflicts_collapses_same_slot_to_one_winner():
    common = "rank(close) * rank(volume)"
    soft = _make_link(expr=common, evidence_type="op_swap", alpha_id="soft", delta_U=0.20)
    hard = _make_link(expr=common, evidence_type="region_swap", alpha_id="hard", delta_U=0.15)
    different_slot = _make_link(expr="rank(returns)", alpha_id="other")
    out = lex_resolve_conflicts([soft, hard, different_slot])
    aliases = [l.alpha_id for l in out]
    assert "hard" in aliases
    assert "soft" not in aliases
    assert "other" in aliases
    assert len(out) == 2


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_under_gates", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class _StubResult:
    sharpe: float = 1.45
    fitness: float = 1.10
    turnover: float = 0.20
    alpha_id: str = "child"
    status: str = "COMPLETE"
    error: Optional[str] = None
    returns: float = 0.0

    def to_dict(self):
        return {
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "alpha_id": self.alpha_id,
            "status": self.status,
            "returns": self.returns,
        }


class _StubSession:
    def __init__(self, result):
        self._result = result

    def login(self):
        return None

    def simulate_and_parse(self, expr, settings, timeout):
        return self._result


def test_cmd_simulate_rejects_when_slot_cooldown_blocks(isolated_artifacts):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried

    module = _load_cli()
    # Seed prior row with low ΔU on the same slot.
    append_tried(
        tried_exprs_path("ut_cooldown_block"),
        expr="rank(close) * rank(volume)",
        sharpe=1.32, fitness=1.00, turnover=0.22,
        alpha_id="prior", status="COMPLETE",
        region="USA", universe="TOP500", decay=6,
        evidence_type="op_swap", altitude=ALTITUDE_L2_OP_FAMILY,
        parent_alpha_id=None, delta_U=0.01,  # below δ_flip
    )

    fake = _StubSession(_StubResult())
    fake_reserve = {"status": "ok", "day": "2026-05-15", "counts": {"simulate": 1}}
    buf = io.StringIO()
    with patch("agent_market.wq_brain.client.session_from_env", return_value=fake), \
            patch("agent_market.wq_brain.quota_monitor.reserve_action",
                  return_value=fake_reserve), \
            patch("agent_market.wq_brain.quota_monitor.release_action",
                  return_value=None), \
            redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        parser = module._build_parser()
        args = parser.parse_args([
            "simulate", "rank(close) * rank(volume)",
            "--tag", "ut_cooldown_block",
            "--region", "USA", "--universe", "TOP500",
            "--auto-persist-sharpe", "999",
        ])
        args.func(args)

    assert excinfo.value.code == 2
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is False
    assert payload["rejected_by"] == "slot_cooldown"
    assert payload["cooldown"]["last_seen_alpha_id"] == "prior"


def test_cmd_simulate_skip_cooldown_overrides_the_gate(isolated_artifacts):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried

    module = _load_cli()
    append_tried(
        tried_exprs_path("ut_cooldown_skip"),
        expr="rank(close) * rank(volume)",
        sharpe=1.32, fitness=1.00, turnover=0.22,
        alpha_id="prior", status="COMPLETE",
        region="USA", universe="TOP500", decay=6,
        evidence_type="op_swap", altitude=ALTITUDE_L2_OP_FAMILY,
        parent_alpha_id=None, delta_U=0.01,
    )

    fake = _StubSession(_StubResult())
    fake_reserve = {"status": "ok", "day": "2026-05-15", "counts": {"simulate": 1}}
    buf = io.StringIO()
    with patch("agent_market.wq_brain.client.session_from_env", return_value=fake), \
            patch("agent_market.wq_brain.quota_monitor.reserve_action",
                  return_value=fake_reserve), \
            patch("agent_market.wq_brain.quota_monitor.release_action",
                  return_value=None), \
            redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        parser = module._build_parser()
        args = parser.parse_args([
            "simulate", "rank(close) * rank(volume)",
            "--tag", "ut_cooldown_skip",
            "--region", "USA", "--universe", "TOP500",
            "--skip-cooldown",
            "--auto-persist-sharpe", "999",
        ])
        args.func(args)
    assert excinfo.value.code == 0
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is True

"""Tests for endpoint probe + failover CLI."""
from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.endpoint_probe import (
    EndpointCandidate,
    ProbeResult,
    _resolve_url,
    first_healthy,
    load_candidates_from_env,
    load_candidates_from_file,
    probe_candidate,
    write_env_local,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "wq_brain.py"


def _fake_resp(payload: dict):
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_resolve_url_normalises_trailing_v1():
    assert _resolve_url("https://example.test/v1") == "https://example.test/v1/chat/completions"
    assert _resolve_url("https://example.test") == "https://example.test/v1/chat/completions"
    assert _resolve_url("https://example.test/v1/") == "https://example.test/v1/chat/completions"


def test_probe_candidate_returns_ok_on_200(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    candidate = EndpointCandidate(base_url="https://example.test",
                                  model="m-good", label="primary")
    with patch("urllib.request.urlopen",
               return_value=_fake_resp({"choices": [], "usage": {}})):
        out = probe_candidate(candidate)
    assert out.ok is True
    assert out.http_status == 200
    assert out.elapsed_ms >= 0


def test_probe_candidate_returns_http_error_on_503(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    candidate = EndpointCandidate(base_url="https://example.test",
                                  model="m-bad")
    http_err = urllib.error.HTTPError(
        url="https://example.test/v1/chat/completions",
        code=503, msg="Service Unavailable",
        hdrs=None, fp=io.BytesIO(b"{\"error\":\"model_not_found\"}"),
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        out = probe_candidate(candidate)
    assert out.ok is False
    assert out.http_status == 503
    assert "model_not_found" in (out.body_excerpt or "")


def test_probe_candidate_missing_credentials_fails(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidate = EndpointCandidate(base_url="https://example.test", model="m")
    out = probe_candidate(candidate)
    assert out.ok is False
    assert "missing" in (out.error or "")


def test_first_healthy_returns_first_ok_probe():
    c_a = EndpointCandidate(base_url="https://a.test", model="m")
    c_b = EndpointCandidate(base_url="https://b.test", model="m")
    probes = [
        ProbeResult(candidate=c_a, ok=False, http_status=503),
        ProbeResult(candidate=c_b, ok=True, http_status=200, elapsed_ms=42),
    ]
    winner = first_healthy(probes)
    assert winner is not None
    assert winner.candidate.base_url == "https://b.test"


def test_first_healthy_none_when_all_fail():
    c = EndpointCandidate(base_url="https://a.test", model="m")
    assert first_healthy([
        ProbeResult(candidate=c, ok=False),
        ProbeResult(candidate=c, ok=False),
    ]) is None


def test_load_candidates_from_env_parses_json(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_ENDPOINTS", json.dumps([
        {"base_url": "https://primary.test/v1", "model": "p1",
         "api_key": "sk-1", "label": "primary"},
        {"base_url": "https://secondary.test/v1", "model": "s1"},
    ]))
    out = load_candidates_from_env()
    assert len(out) == 2
    assert out[0].label == "primary"
    assert out[1].api_key is None


def test_load_candidates_from_env_falls_back_to_triple(monkeypatch):
    monkeypatch.delenv("OPENAI_FALLBACK_ENDPOINTS", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://x.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "m-x")
    out = load_candidates_from_env()
    assert len(out) == 1
    assert out[0].label == "env_default"


def test_load_candidates_from_file_accepts_single_object(tmp_path):
    p = tmp_path / "cand.json"
    p.write_text(json.dumps(
        {"base_url": "https://only.test", "model": "m-only",
         "api_key": "sk-only"}
    ), encoding="utf-8")
    out = load_candidates_from_file(p)
    assert len(out) == 1
    assert out[0].base_url == "https://only.test"


def test_write_env_local_round_trip(tmp_path):
    candidate = EndpointCandidate(
        base_url="https://chosen.test/v1", model="m-good", api_key="sk-good",
    )
    p = tmp_path / ".env.local"
    write_env_local(p, candidate)
    text = p.read_text()
    assert "OPENAI_BASE_URL=https://chosen.test/v1" in text
    assert "OPENAI_API_KEY=sk-good" in text
    assert "OPENAI_MODEL=m-good" in text


def test_write_env_local_preserves_existing_unrelated_keys(tmp_path):
    p = tmp_path / ".env.local"
    p.write_text("CUSTOM_VAR=keep_me\nOPENAI_MODEL=stale\n", encoding="utf-8")
    candidate = EndpointCandidate(
        base_url="https://new.test", model="m-new", api_key="sk-new",
    )
    write_env_local(p, candidate)
    text = p.read_text()
    assert "CUSTOM_VAR=keep_me" in text
    assert "OPENAI_MODEL=m-new" in text  # overwritten
    assert "stale" not in text


# ── CLI integration ──────────────────────────────────────────────────────


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_endpoint", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_endpoint_failover_cli_writes_env_local_on_first_healthy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    cand_file = tmp_path / "candidates.json"
    cand_file.write_text(json.dumps([
        {"base_url": "https://bad.test", "model": "bad", "label": "bad"},
        {"base_url": "https://good.test", "model": "good", "label": "good"},
    ]), encoding="utf-8")

    bad_err = urllib.error.HTTPError(
        url="https://bad.test/v1/chat/completions", code=503,
        msg="Service Unavailable", hdrs=None,
        fp=io.BytesIO(b'{"error":"unavailable"}'),
    )
    good_resp = _fake_resp({"choices": [], "usage": {}})
    sequence = [bad_err, good_resp]

    def fake_urlopen(req, timeout=None):
        nxt = sequence.pop(0)
        if isinstance(nxt, urllib.error.HTTPError):
            raise nxt
        return nxt

    # Pretend the repo root is the tmp dir so .env.local lands there.
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
            patch("agent_market.wq_brain.paths.repo_root",
                  return_value=tmp_path):
        module = _load_cli()
        parser = module._build_parser()
        args = parser.parse_args([
            "endpoint", "failover",
            "--candidates-file", str(cand_file),
            "--timeout", "5",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
            args.func(args)

    assert excinfo.value.code == 0
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is True
    assert payload["chosen"]["label"] == "good"
    assert (tmp_path / ".env.local").exists()
    assert "OPENAI_BASE_URL=https://good.test" in (tmp_path / ".env.local").read_text()


def test_endpoint_failover_cli_returns_error_when_all_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    cand_file = tmp_path / "candidates.json"
    cand_file.write_text(json.dumps([
        {"base_url": "https://dead.test", "model": "dead"},
    ]), encoding="utf-8")
    err = urllib.error.HTTPError(
        url="https://dead.test/v1/chat/completions", code=503,
        msg="bad", hdrs=None, fp=io.BytesIO(b"{}"),
    )
    with patch("urllib.request.urlopen", side_effect=err), \
            patch("agent_market.wq_brain.paths.repo_root",
                  return_value=tmp_path):
        module = _load_cli()
        parser = module._build_parser()
        args = parser.parse_args([
            "endpoint", "failover", "--candidates-file", str(cand_file),
            "--timeout", "5",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
            args.func(args)
    assert excinfo.value.code == 1
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is False
    assert payload["error"] == "no healthy candidate"

"""Tests for the ``ping-llm`` preflight subcommand."""
from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "wq_brain.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "_wq_brain_cli_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_cli_module()


def _fake_urlopen_response(payload: dict) -> MagicMock:
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_ping_llm_emits_ok_on_successful_response(cli_module, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "MiniMax-M2.7-highspeed")

    fake = _fake_urlopen_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    )
    buf = io.StringIO()
    with patch("urllib.request.urlopen", return_value=fake), \
            redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        parser = cli_module._build_parser()
        args = parser.parse_args(["ping-llm", "--timeout", "5"])
        args.func(args)

    assert excinfo.value.code == 0
    out = json.loads(buf.getvalue())
    assert out["ok"] is True
    assert out["model"] == "MiniMax-M2.7-highspeed"
    assert out["prompt_tokens"] == 3
    assert out["total_tokens"] == 4
    assert out["first_choice_role"] == "assistant"
    assert out["url"].endswith("/v1/chat/completions")


def test_ping_llm_emits_http_status_on_http_error(cli_module, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "MiniMax-M2.7-highspeed")

    import urllib.error
    http_err = urllib.error.HTTPError(
        url="https://example.test/v1/chat/completions",
        code=502,
        msg="Bad Gateway",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"upstream timeout"}'),
    )
    buf = io.StringIO()
    with patch("urllib.request.urlopen", side_effect=http_err), \
            redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        parser = cli_module._build_parser()
        args = parser.parse_args(["ping-llm"])
        args.func(args)

    assert excinfo.value.code == 1
    out = json.loads(buf.getvalue())
    assert out["ok"] is False
    assert out["http_status"] == 502
    assert "upstream timeout" in out["body"]


def test_ping_llm_strips_trailing_v1_from_base_url(cli_module, monkeypatch):
    """When OPENAI_BASE_URL already ends in /v1 we must not append /v1 again."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "MiniMax-M2.7-highspeed")

    captured_urls: list[str] = []

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        return _fake_urlopen_response({"choices": [], "usage": {}})

    buf = io.StringIO()
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
            redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        parser = cli_module._build_parser()
        args = parser.parse_args(["ping-llm"])
        args.func(args)

    assert excinfo.value.code == 0
    assert captured_urls == ["https://example.test/v1/chat/completions"]


def test_ping_llm_appends_v1_when_base_url_is_host_root(cli_module, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "MiniMax-M2.7-highspeed")

    captured_urls: list[str] = []

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        return _fake_urlopen_response({"choices": [], "usage": {}})

    buf = io.StringIO()
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
            redirect_stdout(buf), pytest.raises(SystemExit):
        parser = cli_module._build_parser()
        args = parser.parse_args(["ping-llm"])
        args.func(args)

    assert captured_urls == ["https://example.test/v1/chat/completions"]


def test_ping_llm_emits_missing_env_error(cli_module, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        parser = cli_module._build_parser()
        args = parser.parse_args(["ping-llm"])
        args.func(args)

    assert excinfo.value.code == 2
    out = json.loads(buf.getvalue())
    assert out["ok"] is False
    assert "OPENAI_BASE_URL" in out["error"]

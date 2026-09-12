from __future__ import annotations

from unittest.mock import Mock


def test_auto_improver_uses_openai_model_and_env(monkeypatch) -> None:
    from ws_production.auto_improver import AutoImprover

    monkeypatch.setenv("LLM_BASE_URL", "http://proxy.internal:4141/v1")
    monkeypatch.setenv("LLM_API_KEY", "_")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_MODEL", raising=False)

    proc = Mock()
    proc.returncode = 0
    proc.stdout = '{"type":"text","part":{"text":"OK"}}\n'

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env") or {}
        return proc

    monkeypatch.setattr("ws_production.auto_improver.subprocess.run", _fake_run)

    improver = AutoImprover()
    text = improver._llm_call("", "Reply with OK")

    assert improver.opencode_model == "custom/gpt-5.2"
    assert improver.model == "gpt-5.2"
    assert improver.base_url == "http://proxy.internal:4141/v1"
    assert text == "OK"
    assert captured["cmd"][:4] == ["opencode", "run", "-m", improver.opencode_model]
    assert captured["env"]["OPENAI_BASE_URL"] == "http://proxy.internal:4141/v1"
    assert captured["env"]["OPENAI_API_KEY"] == "_"
    assert captured["env"]["OPENCODE_CONFIG"].endswith("/.opencode.json")


def test_project_opencode_config_uses_chat_compatible_provider() -> None:
    import json
    from pathlib import Path

    payload = json.loads((Path(__file__).resolve().parents[1] / ".opencode.json").read_text(encoding="utf-8"))

    assert payload["model"].startswith("custom/")
    assert payload["small_model"] == payload["model"]
    model_name = payload["model"].split("/", 1)[1]
    assert model_name in payload["provider"]["custom"]["models"]
    assert payload["provider"]["custom"]["npm"] == "@ai-sdk/openai-compatible"

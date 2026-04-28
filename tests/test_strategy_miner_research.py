from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _reset_opencli_cache(research) -> None:  # noqa: ANN001
    research._OPENCLI_AVAILABLE = None
    research._OPENCLI_COMMAND = None
    research._OPENCLI_PATH_PREFIX = None


def test_opencli_available_finds_user_local_install(monkeypatch, tmp_path: Path) -> None:
    from agent_market.strategy_miner import research

    _reset_opencli_cache(research)

    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    opencli_path = bin_dir / "opencli"
    opencli_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    opencli_path.chmod(0o755)

    calls = []

    def _fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        calls.append((cmd, kwargs.get("env", {})))
        return SimpleNamespace(returncode=0, stdout="1.6.8\n", stderr="")

    monkeypatch.setattr(research, "_scrubbed_env", lambda: {"PATH": "/usr/bin:/bin"})
    monkeypatch.setattr(research.Path, "home", lambda: home)
    monkeypatch.setattr(research.shutil, "which", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(research.subprocess, "run", _fake_run)

    assert research._opencli_available() is True
    assert research._OPENCLI_COMMAND == [str(opencli_path)]
    assert research._OPENCLI_PATH_PREFIX == str(bin_dir)
    assert calls[0][0] == [str(opencli_path), "--version"]
    assert calls[0][1]["PATH"].startswith(f"{bin_dir}:")


def test_run_opencli_uses_resolved_binary_and_path_prefix(monkeypatch) -> None:
    from agent_market.strategy_miner import research

    _reset_opencli_cache(research)
    research._OPENCLI_AVAILABLE = True
    research._OPENCLI_COMMAND = ["/home/test/.local/bin/opencli"]
    research._OPENCLI_PATH_PREFIX = "/home/test/.local/bin"

    captured = {}

    def _fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return SimpleNamespace(returncode=0, stdout='[{"title":"ok"}]', stderr="")

    monkeypatch.setattr(research, "_scrubbed_env", lambda: {"PATH": "/usr/bin:/bin"})
    monkeypatch.setattr(research.subprocess, "run", _fake_run)

    raw = research._run_opencli(["arxiv", "search", "btc"])

    assert raw == '[{"title":"ok"}]'
    assert captured["cmd"] == ["/home/test/.local/bin/opencli", "arxiv", "search", "btc", "--format", "json"]
    assert captured["env"]["PATH"].startswith("/home/test/.local/bin:")


def test_opencli_available_warns_when_missing(monkeypatch) -> None:
    from agent_market.strategy_miner import research

    _reset_opencli_cache(research)
    messages: list[str] = []

    monkeypatch.setattr(research, "_iter_opencli_candidates", lambda: [])
    monkeypatch.setattr(research.logger, "warning", lambda message, *args: messages.append(message % args if args else message))

    assert research._opencli_available() is False
    assert messages == ["opencli not found in PATH or common install paths — web research disabled"]

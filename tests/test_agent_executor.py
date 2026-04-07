from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

import agent_market.agents.executor as executor_mod
from agent_market.agents.executor import OpenAIChatExecutor, OpenCodeExecutor
from runner_fsm.dtypes import AgentResult


class _FakeResponse:
    def __init__(self, chunks: list[bytes], *, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.encoding = "utf-8"
        self._chunks = list(chunks)
        self.closed = False

    def iter_content(self, chunk_size: int = 65536):  # noqa: ARG002
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


def test_openai_executor_streaming_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "OK"},
            }
        ],
        "usage": {"total_tokens": 7, "prompt_tokens": 5, "completion_tokens": 2},
    }
    response = _FakeResponse([json.dumps(payload).encode("utf-8")])

    def _fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return response

    monkeypatch.setattr(executor_mod.requests, "post", _fake_post)
    exe = OpenAIChatExecutor(
        base_url="http://example.test/v1",
        api_key="dummy",
        model="gpt-test",
        timeout_seconds=5,
        retries=0,
    )
    result = exe.run("ping")
    assert result.assistant_text == "OK"
    assert result.usage == {"total_tokens": 7, "prompt_tokens": 5, "completion_tokens": 2}
    assert response.closed is True


def test_openai_executor_respects_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_LLM_REQUEST_TIMEOUT", "7")
    monkeypatch.setenv("AGENT_MARKET_LLM_MAX_CONCURRENT", "3")
    monkeypatch.setenv("AGENT_MARKET_LLM_SLOT_WAIT_SECONDS", "90")
    exe = OpenAIChatExecutor(
        base_url="http://example.test/v1",
        api_key="dummy",
        model="gpt-test",
        timeout_seconds=60,
        retries=0,
    )
    assert exe._timeout == 7.0
    assert exe._max_concurrent == 3
    assert exe._slot_wait_timeout == 90.0


def test_openai_executor_wall_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse([b"{", b"\"choices\":[", b"]}"])

    def _fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return response

    clock = {"now": 0.0}

    def _fake_monotonic() -> float:
        clock["now"] += 0.03
        return clock["now"]

    monkeypatch.setattr(executor_mod.requests, "post", _fake_post)
    monkeypatch.setattr(executor_mod.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_mod.time, "monotonic", _fake_monotonic)

    exe = OpenAIChatExecutor(
        base_url="http://example.test/v1",
        api_key="dummy",
        model="gpt-test",
        timeout_seconds=0.05,
        retries=0,
    )
    with pytest.raises(RuntimeError, match="llm_wall_timeout"):
        exe.run("slow")
    assert response.closed is True


def test_openai_executor_slot_wait_does_not_consume_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "OK"},
            }
        ],
    }
    response = _FakeResponse([json.dumps(payload).encode("utf-8")])

    def _fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return response

    clock = {"now": 0.0}

    def _fake_monotonic() -> float:
        return clock["now"]

    @contextmanager
    def _fake_slot(*args, **kwargs):  # noqa: ANN002, ANN003
        clock["now"] += 10.0
        yield

    monkeypatch.setattr(executor_mod.requests, "post", _fake_post)
    monkeypatch.setattr(executor_mod.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_mod.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(executor_mod, "_llm_request_slot", _fake_slot)

    exe = OpenAIChatExecutor(
        base_url="http://example.test/v1",
        api_key="dummy",
        model="gpt-test",
        timeout_seconds=1.0,
        retries=0,
    )
    result = exe.run("queued")
    assert result.assistant_text == "OK"


def test_opencode_executor_extracts_usage_from_tool_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeOpenCodeClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

        def run(self, prompt: str, *, on_turn=None):  # noqa: ANN001, ARG002
            return AgentResult(
                assistant_text="done",
                raw={"kind": "fake"},
                tool_trace=[
                    {
                        "turn": 1,
                        "tokens": {
                            "input": 100,
                            "output": 20,
                            "reasoning": 5,
                            "cache_read": 10,
                            "cache_write": 0,
                            "total": 125,
                            "cost": 0.15,
                        },
                    },
                    {
                        "turn": 2,
                        "tokens": {
                            "input": 40,
                            "output": 10,
                            "reasoning": 0,
                            "cache_read": 5,
                            "cache_write": 0,
                            "total": 50,
                            "cost": 0.05,
                        },
                    },
                ],
            )

        def close(self) -> None:
            return

    monkeypatch.setattr("runner_fsm.opencode.client.OpenCodeClient", _FakeOpenCodeClient)
    exe = OpenCodeExecutor(repo=Path("."), model="github-copilot/gpt-5-mini")
    result = exe.run("hello")
    assert result.assistant_text == "done"
    assert result.usage == {
        "total_tokens": 175,
        "prompt_tokens": 140,
        "completion_tokens": 30,
        "reasoning_tokens": 5,
        "cache_read_tokens": 15,
        "cache_write_tokens": 0,
        "cost_usd": 0.2,
    }

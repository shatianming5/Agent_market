"""Tests for runner_fsm module: dtypes, security, tool_parser, client helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------


def test_cmd_result_immutable():
    from runner_fsm.dtypes import CmdResult

    r = CmdResult(cmd="echo hi", rc=0, stdout="hi\n", stderr="")
    assert r.rc == 0
    assert r.timed_out is False
    with pytest.raises(AttributeError):
        r.rc = 1  # type: ignore[misc]


def test_turn_event_defaults():
    from runner_fsm.dtypes import TurnEvent

    ev = TurnEvent(turn=1)
    assert ev.assistant_text == ""
    assert ev.calls == ()
    assert ev.results == ()
    assert ev.finished is False


def test_turn_event_with_data():
    from runner_fsm.dtypes import TurnEvent

    ev = TurnEvent(turn=3, assistant_text="hello", finished=True)
    assert ev.turn == 3
    assert ev.assistant_text == "hello"
    assert ev.finished is True


def test_agent_result():
    from runner_fsm.dtypes import AgentResult

    r = AgentResult(assistant_text="done", raw={"key": "val"})
    assert r.assistant_text == "done"
    assert r.raw == {"key": "val"}
    assert r.tool_trace is None


# ---------------------------------------------------------------------------
# security: cmd_allowed
# ---------------------------------------------------------------------------


def test_cmd_allowed_normal_commands():
    from runner_fsm.utils.security import cmd_allowed

    ok, reason = cmd_allowed("ls -la")
    assert ok is True
    assert reason is None

    ok, reason = cmd_allowed("python train.py --config cfg.yaml")
    assert ok is True


def test_cmd_allowed_blocks_rm_rf_root():
    from runner_fsm.utils.security import cmd_allowed

    ok, reason = cmd_allowed("rm -rf /")
    assert ok is False
    assert "hard_deny" in reason

    ok, reason = cmd_allowed("rm -rf /*")
    assert ok is False


def test_cmd_allowed_blocks_rm_rf_home():
    from runner_fsm.utils.security import cmd_allowed

    ok, reason = cmd_allowed("rm -rf ~")
    assert ok is False

    ok, reason = cmd_allowed("rm -rf ~/")
    assert ok is False

    ok, reason = cmd_allowed("rm -rf $HOME/")
    assert ok is False


def test_cmd_allowed_blocks_fork_bomb():
    from runner_fsm.utils.security import cmd_allowed

    ok, reason = cmd_allowed(":(){ :|:& };:")
    assert ok is False
    assert "hard_deny" in reason


def test_cmd_allowed_blocks_sudo():
    from runner_fsm.utils.security import cmd_allowed

    ok, reason = cmd_allowed("sudo apt install foo")
    assert ok is False
    assert "safe_deny" in reason


def test_cmd_allowed_blocks_dangerous_commands():
    from runner_fsm.utils.security import cmd_allowed

    for cmd in ["shutdown -h now", "reboot", "mkfs /dev/sda1"]:
        ok, reason = cmd_allowed(cmd)
        assert ok is False, f"Should block: {cmd}"


def test_cmd_allowed_empty():
    from runner_fsm.utils.security import cmd_allowed

    ok, reason = cmd_allowed("")
    assert ok is False
    assert reason == "empty_command"

    ok, reason = cmd_allowed("   ")
    assert ok is False


# ---------------------------------------------------------------------------
# security: safe_env
# ---------------------------------------------------------------------------


def test_safe_env_strict():
    from runner_fsm.utils.security import safe_env

    env = safe_env({"PATH": "/usr/bin"}, {"MY_VAR": "val"}, unattended="strict")
    assert env["PATH"] == "/usr/bin"
    assert env["MY_VAR"] == "val"
    assert env["CI"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_safe_env_non_strict():
    from runner_fsm.utils.security import safe_env

    env = safe_env({}, {}, unattended="permissive")
    assert "CI" not in env


# ---------------------------------------------------------------------------
# security: looks_interactive
# ---------------------------------------------------------------------------


def test_looks_interactive():
    from runner_fsm.utils.security import looks_interactive

    assert looks_interactive("docker login") is True
    assert looks_interactive("docker login --password-stdin") is False
    assert looks_interactive("echo hello") is False
    assert looks_interactive("") is False


# ---------------------------------------------------------------------------
# tool_parser: parse_tool_calls
# ---------------------------------------------------------------------------


def test_parse_bash_tag():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = '<bash command="ls -la" description="list files"/>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    assert calls[0].payload["command"] == "ls -la"


def test_parse_read_tag():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = '<read filePath="/tmp/test.py"/>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    assert calls[0].payload["filePath"] == "/tmp/test.py"


def test_parse_write_tag_with_body():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = '<write filePath="/tmp/out.txt">hello world</write>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].payload["filePath"] == "/tmp/out.txt"
    assert calls[0].payload["content"] == "hello world"


def test_parse_edit_tag():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = '<edit filePath="/tmp/f.py" oldString="foo" newString="bar"/>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].payload["oldString"] == "foo"
    assert calls[0].payload["newString"] == "bar"


def test_parse_multiple_calls():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = """
    <bash command="echo hello"/>
    <read filePath="/etc/hosts"/>
    <bash command="echo world"/>
    """
    calls = parse_tool_calls(text)
    assert len(calls) == 3
    assert calls[0].kind == "bash"
    assert calls[1].kind == "file"
    assert calls[2].kind == "bash"


def test_parse_tool_call_json_format():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = '<tool_call>{"command": "pwd"}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    assert calls[0].payload["command"] == "pwd"


def test_parse_no_tool_calls():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    calls = parse_tool_calls("just plain text with no tools")
    assert calls == []


def test_parse_deduplicates():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = '<bash command="ls"/><bash command="ls"/>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1


def test_parse_xml_unescape():
    from runner_fsm.opencode.tool_parser import parse_tool_calls

    text = '<bash command="echo &amp;hello&lt;world&gt;"/>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert "&hello<world>" in calls[0].payload["command"]


# ---------------------------------------------------------------------------
# tool_parser: format_tool_results
# ---------------------------------------------------------------------------


def test_format_tool_results():
    from runner_fsm.opencode.tool_parser import ToolResult, format_tool_results

    results = [
        ToolResult(kind="bash", ok=True, detail={"stdout": "hello\n", "rc": 0}),
        ToolResult(kind="file", ok=True, detail={"filePath": "/tmp/f.py", "content": "x=1"}),
    ]
    text = format_tool_results(results)
    assert "tool_result" in text
    parsed = json.loads(text.split("```tool_result\n")[1].split("\n```")[0])
    assert len(parsed) == 2
    assert parsed[0]["tool"] == "bash"
    assert parsed[0]["ok"] is True


# ---------------------------------------------------------------------------
# client helpers
# ---------------------------------------------------------------------------


def test_safe_int():
    from runner_fsm.opencode.client import _safe_int

    assert _safe_int("42", 0) == 42
    assert _safe_int("bad", 5) == 5
    assert _safe_int(None, 3) == 3


def test_safe_float():
    from runner_fsm.opencode.client import _safe_float

    assert _safe_float("3.14", 0.0) == pytest.approx(3.14)
    assert _safe_float("bad", 1.5) == 1.5
    assert _safe_float(None, 2.0) == 2.0


def test_find_free_port():
    from runner_fsm.opencode.client import _find_free_port

    port = _find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_opencode_request_error():
    from runner_fsm.opencode.client import OpenCodeRequestError

    err = OpenCodeRequestError(method="POST", url="http://localhost/msg", status=500, detail="server error")
    assert err.method == "POST"
    assert err.status == 500
    assert "500" in str(err)
    assert "server error" in str(err)


def test_stale_thinking_timeout():
    from runner_fsm.opencode.client import StaleThinkingTimeout

    err = StaleThinkingTimeout("timeout after 180s")
    assert isinstance(err, RuntimeError)
    assert "180s" in str(err)


# ---------------------------------------------------------------------------
# client: _is_transport_error
# ---------------------------------------------------------------------------


def test_is_transport_error_with_status():
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    e = OpenCodeRequestError(method="POST", url="http://x", status=500, detail="internal")
    assert OpenCodeClient._is_transport_error(e) is False


def test_is_transport_error_connection_refused():
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    e = OpenCodeRequestError(method="POST", url="http://x", status=None, detail="Connection refused")
    assert OpenCodeClient._is_transport_error(e) is True


def test_is_transport_error_timeout():
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    e = OpenCodeRequestError(method="POST", url="http://x", status=None, detail="request timed out")
    assert OpenCodeClient._is_transport_error(e) is True


def test_is_transport_error_connection_reset():
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    e = OpenCodeRequestError(method="POST", url="http://x", status=None, detail="Connection reset by peer")
    assert OpenCodeClient._is_transport_error(e) is True


def test_is_transport_error_dns():
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    e = OpenCodeRequestError(method="POST", url="http://x", status=None, detail="Name or service not known")
    assert OpenCodeClient._is_transport_error(e) is True


def test_is_transport_error_other():
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    e = OpenCodeRequestError(method="POST", url="http://x", status=None, detail="some random error")
    assert OpenCodeClient._is_transport_error(e) is False


# ---------------------------------------------------------------------------
# client: retry logic (_post_message_with_retry) - mocked
# ---------------------------------------------------------------------------


def test_retry_succeeds_on_second_attempt():
    """Retry should succeed when second attempt passes."""
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    client = object.__new__(OpenCodeClient)
    client._request_retry_attempts = 2
    client._request_retry_backoff_seconds = 0.0
    client._session_recover_attempts = 0
    client._session_recover_backoff_seconds = 0.0
    client._owns_local_server = False
    client._context_length = None
    client._stale_timeout = 10.0
    client._stale_timeout_event = None

    call_count = 0

    def mock_post_message(*, model, text, include_context=True):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OpenCodeRequestError(method="POST", url="http://x/msg", status=500, detail="transient")
        return {"text": "ok"}

    client._post_message = mock_post_message
    result = client._post_message_with_retry(model="test", text="hello")
    assert result == {"text": "ok"}
    assert call_count == 2


def test_retry_exhausted_raises():
    """Should raise after all retries are exhausted."""
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    client = object.__new__(OpenCodeClient)
    client._request_retry_attempts = 1
    client._request_retry_backoff_seconds = 0.0
    client._session_recover_attempts = 0
    client._session_recover_backoff_seconds = 0.0
    client._owns_local_server = False
    client._context_length = None
    client._stale_timeout = 10.0
    client._stale_timeout_event = None

    def mock_post_message(*, model, text, include_context=True):
        raise OpenCodeRequestError(method="POST", url="http://x/msg", status=502, detail="bad gateway")

    client._post_message = mock_post_message
    with pytest.raises(OpenCodeRequestError) as exc_info:
        client._post_message_with_retry(model="test", text="hello")
    assert exc_info.value.status == 502


def test_retry_no_retry_on_400():
    """400 errors should NOT be retried (not in retryable status list)."""
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    client = object.__new__(OpenCodeClient)
    client._request_retry_attempts = 3
    client._request_retry_backoff_seconds = 0.0
    client._session_recover_attempts = 0
    client._session_recover_backoff_seconds = 0.0
    client._owns_local_server = False
    client._context_length = None
    client._stale_timeout = 10.0
    client._stale_timeout_event = None

    call_count = 0

    def mock_post_message(*, model, text, include_context=True):
        nonlocal call_count
        call_count += 1
        raise OpenCodeRequestError(method="POST", url="http://x/msg", status=400, detail="bad request")

    client._post_message = mock_post_message
    with pytest.raises(OpenCodeRequestError):
        client._post_message_with_retry(model="test", text="hello")
    assert call_count == 1  # no retries for 400


def test_retry_on_429_rate_limit():
    """429 (rate limit) should be retried."""
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError

    client = object.__new__(OpenCodeClient)
    client._request_retry_attempts = 2
    client._request_retry_backoff_seconds = 0.0
    client._session_recover_attempts = 0
    client._session_recover_backoff_seconds = 0.0
    client._owns_local_server = False
    client._context_length = None
    client._stale_timeout = 10.0
    client._stale_timeout_event = None

    call_count = 0

    def mock_post_message(*, model, text, include_context=True):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OpenCodeRequestError(method="POST", url="http://x/msg", status=429, detail="rate limited")
        return {"text": "ok"}

    client._post_message = mock_post_message
    result = client._post_message_with_retry(model="test", text="hello")
    assert result == {"text": "ok"}
    assert call_count == 3


def test_stale_timeout_propagated():
    """Stale thinking timeout should be raised immediately, no retry."""
    import threading
    from runner_fsm.opencode.client import OpenCodeClient, OpenCodeRequestError, StaleThinkingTimeout

    client = object.__new__(OpenCodeClient)
    client._request_retry_attempts = 3
    client._request_retry_backoff_seconds = 0.0
    client._session_recover_attempts = 0
    client._session_recover_backoff_seconds = 0.0
    client._owns_local_server = False
    client._context_length = None
    client._stale_timeout = 10.0
    client._stale_timeout_event = threading.Event()
    client._stale_timeout_event.set()  # simulate stale timeout triggered

    def mock_post_message(*, model, text, include_context=True):
        raise OpenCodeRequestError(method="POST", url="http://x/msg", status=None, detail="timeout")

    client._post_message = mock_post_message
    with pytest.raises(StaleThinkingTimeout):
        client._post_message_with_retry(model="test", text="hello")


def test_opencode_run_on_turn_uses_wait_monitor(tmp_path):
    """Regression: on_turn should use the module monitor, not a missing instance attr."""
    from runner_fsm.opencode.client import OpenCodeClient

    client = object.__new__(OpenCodeClient)
    client._repo = tmp_path
    client._tool_policy = None
    client._unattended = "strict"
    client._max_turns = 1
    client._token_log_path = None
    client._stale_timeout = 10.0
    client._proc = None
    client._model_obj = {"providerID": "openai", "modelID": "unit"}
    client._model_str = "openai/unit"
    client._extract_context_info = lambda msg: None
    client._print_context_status = lambda *args, **kwargs: None
    client._post_message_with_retry = lambda *, model, text: {
        "parts": [{"type": "text", "text": "done"}],
    }

    events = []
    result = client.run("hello", on_turn=events.append)

    assert result.assistant_text == "done"
    assert len(events) == 1
    assert events[0].finished is True


# ---------------------------------------------------------------------------
# OpenCodeServerConfig
# ---------------------------------------------------------------------------


def test_server_config_frozen():
    from runner_fsm.opencode.client import OpenCodeServerConfig

    cfg = OpenCodeServerConfig(base_url="http://localhost:8080", username="user", password="pass")
    assert cfg.base_url == "http://localhost:8080"
    with pytest.raises(AttributeError):
        cfg.base_url = "http://other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# tool_executor: _is_env_like, _within_root
# ---------------------------------------------------------------------------


def test_is_env_like():
    from runner_fsm.opencode.tool_executor import _is_env_like

    assert _is_env_like(Path(".env")) is True
    assert _is_env_like(Path(".env.local")) is True
    assert _is_env_like(Path("prod.env")) is True
    assert _is_env_like(Path("config.yaml")) is False
    assert _is_env_like(Path("train.py")) is False


def test_within_root():
    from runner_fsm.opencode.tool_executor import _within_root

    root = Path("/project")
    assert _within_root(root, Path("/project/src/main.py")) is True
    assert _within_root(root, Path("/other/file.py")) is False


# ---------------------------------------------------------------------------
# tool_executor: ToolPolicy enforcement
# ---------------------------------------------------------------------------


def test_tool_policy_blocks_disallowed_kind(tmp_path: Path):
    from runner_fsm.opencode.tool_executor import ToolPolicy, execute_tool_calls
    from runner_fsm.opencode.tool_parser import ToolCall

    policy = ToolPolicy(
        repo=tmp_path,
        unattended="strict",
        allowed_tool_kinds=frozenset({"file"}),
        bash_allow=True,
        bash_allowlist=(),
        bash_timeout_seconds=60,
    )
    calls = [ToolCall(kind="bash", start=0, payload={"command": "echo hi"})]
    results = execute_tool_calls(calls, policy=policy)
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].detail.get("error") == "tool_not_allowed"


def test_tool_policy_bash_allowlist_prefix(tmp_path: Path):
    from runner_fsm.opencode.tool_executor import ToolPolicy, execute_tool_calls
    from runner_fsm.opencode.tool_parser import ToolCall

    policy = ToolPolicy(
        repo=tmp_path,
        unattended="strict",
        allowed_tool_kinds=None,
        bash_allow=True,
        bash_allowlist=("echo ",),
        bash_timeout_seconds=60,
    )
    calls = [ToolCall(kind="bash", start=0, payload={"command": "ls"})]
    results = execute_tool_calls(calls, policy=policy)
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].detail.get("error") == "bash_not_in_allowlist"


def test_tool_policy_bash_timeout_enforced(tmp_path: Path):
    from runner_fsm.opencode.tool_executor import ToolPolicy, execute_tool_calls
    from runner_fsm.opencode.tool_parser import ToolCall

    policy = ToolPolicy(
        repo=tmp_path,
        unattended="strict",
        allowed_tool_kinds=None,
        bash_allow=True,
        bash_allowlist=("python3 ",),
        bash_timeout_seconds=1,
    )
    calls = [
        ToolCall(
            kind="bash",
            start=0,
            payload={"command": 'python3 -c "import time; time.sleep(2)"'},
        )
    ]
    results = execute_tool_calls(calls, policy=policy)
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].detail.get("rc") == 124
    assert results[0].detail.get("timed_out") is True


def test_tool_policy_blocks_file_outside_repo(tmp_path: Path):
    from runner_fsm.opencode.tool_executor import ToolPolicy, execute_tool_calls
    from runner_fsm.opencode.tool_parser import ToolCall

    policy = ToolPolicy(repo=tmp_path, unattended="strict")
    calls = [ToolCall(kind="file", start=0, payload={"filePath": "/etc/hosts"})]
    results = execute_tool_calls(calls, policy=policy)
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].detail.get("error") == "path_outside_repo"


def test_tool_policy_blocks_env_file_write(tmp_path: Path):
    from runner_fsm.opencode.tool_executor import ToolPolicy, execute_tool_calls
    from runner_fsm.opencode.tool_parser import ToolCall

    policy = ToolPolicy(repo=tmp_path, unattended="strict")
    calls = [ToolCall(kind="file", start=0, payload={"filePath": ".env", "content": "X=1"})]
    results = execute_tool_calls(calls, policy=policy)
    assert len(results) == 1
    assert results[0].ok is False
    assert "env_files" in str(results[0].detail.get("error") or "")

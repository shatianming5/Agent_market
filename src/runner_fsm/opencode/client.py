from __future__ import annotations

import base64
import copy
import json
import os
import signal
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..dtypes import AgentClient, AgentResult, TurnEvent
from .tool_parser import parse_tool_calls, format_tool_results
from .tool_executor import ToolPolicy, execute_tool_calls
from ..utils.subprocess import tail

# Re-export from _types for backward compatibility
from ._types import (  # noqa: F401
    OpenCodeRequestError,
    OpenCodeServerConfig,
    StaleThinkingTimeout,
    _active_clients,
    _find_free_port,
    _rprint,
    _safe_float,
    _safe_int,
)
from ._monitor import LLMWaitMonitor as _LLMWaitMonitor

class OpenCodeClient(AgentClient):

    def __init__(
        self,
        *,
        repo: Path,
        model: str,
        base_url: str | None,
        timeout_seconds: int,
        request_retry_attempts: int = 2,
        request_retry_backoff_seconds: float = 2.0,
        session_recover_attempts: int | None = None,
        session_recover_backoff_seconds: float | None = None,
        context_length: int | None = None,
        max_prompt_chars: int | None = None,
        unattended: str,
        max_turns: int = 20,
        server_log_path: Path | None = None,
        username: str | None = None,
        password: str | None = None,
        session_title: str | None = None,
        stale_timeout: float = 180.0,
        auto_compact: bool | None = None,
        permission_overrides: dict[str, Any] | None = None,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        self._repo = repo
        self._timeout_seconds = int(timeout_seconds or 300)
        self._request_retry_attempts = max(0, int(request_retry_attempts or 0))
        self._request_retry_backoff_seconds = max(0.0, float(request_retry_backoff_seconds or 2.0))
        self._session_recover_attempts = max(0, _safe_int(
            session_recover_attempts if session_recover_attempts is not None
            else os.environ.get("OPENCODE_SESSION_RECOVER_ATTEMPTS", "2"),
            2,
        ))
        self._session_recover_backoff_seconds = max(0.0, _safe_float(
            session_recover_backoff_seconds if session_recover_backoff_seconds is not None
            else os.environ.get("OPENCODE_SESSION_RECOVER_BACKOFF_SECONDS", "2.0"),
            2.0,
        ))
        self._context_length = int(context_length or 0) or None
        self._max_prompt_chars = int(max_prompt_chars or 0) or None
        self._max_turns = max(1, int(max_turns or 20))
        self._unattended = str(unattended or "strict").strip().lower() or "strict"
        self._session_title = str(session_title or f"runner:{repo.name}")
        self._server_log_path = server_log_path.resolve() if server_log_path is not None else None
        self._permission_overrides = permission_overrides or {}
        self._tool_policy = tool_policy

        model_str = str(model or "").strip()
        if not model_str:
            provider_id, model_id = "openai", "gpt-4o-mini"
        elif "/" in model_str:
            provider_id, model_id = model_str.split("/", 1)
            provider_id = provider_id.strip() or "openai"
            model_id = model_id.strip() or "gpt-4o-mini"
        else:
            provider_id, model_id = "openai", model_str
        self._model_obj: dict[str, str] = {"providerID": provider_id, "modelID": model_id}
        self._model_str: str = f"{provider_id}/{model_id}"

        self._stale_timeout = max(0.0, float(stale_timeout or 180.0))
        self._auto_compact = auto_compact

        self._proc: subprocess.Popen[str] | None = None
        self._proxy_proc: subprocess.Popen[str] | None = None
        self._proxy_log_file = None
        self._token_log_path: Path | None = None
        self._temp_config_home: Path | None = None
        self._server_log_file = None
        self._owns_local_server = not bool(base_url)
        _active_clients.add(self)

        if base_url:
            base_url_s = str(base_url).strip()
            if not base_url_s:
                raise ValueError("empty_url")
            self._server = OpenCodeServerConfig(
                base_url=base_url_s.rstrip("/"),
                username=(username or "opencode").strip() or "opencode",
                password=(password or "").strip(),
            )
        else:
            self._server = self._start_local_server(
                repo=repo, server_log_path=self._server_log_path, username=username
            )

        try:
            self._wait_health()
            if self._auto_compact is not None:
                try:
                    self._request_json("PATCH", "/global/config",
                                       body={"compaction": {"auto": bool(self._auto_compact)}},
                                       require_auth=bool(self._server.password))
                except Exception as exc:
                    print(f"    [diag] auto_compact config update failed: {exc}", flush=True)
            self._create_session()
        except Exception:
            try:
                self.close()
            except Exception as close_exc:
                print(f"    [diag] cleanup during init failure also failed: {close_exc}", flush=True)
            raise

    def close(self) -> None:
        _active_clients.discard(self)
        self._stop_local_server()

    @staticmethod
    def _extract_context_info(msg: Any) -> dict[str, Any] | None:
        """Extract token usage and compaction events from a message response.

        Returns a dict with keys: input, output, reasoning, cache_read, cache_write,
        total, cost, compaction (bool), summary (bool).
        Returns None if the response doesn't contain token info (graceful degradation).
        """
        if not isinstance(msg, dict):
            return None
        info = msg.get("info")
        if not isinstance(info, dict):
            return None
        tokens = info.get("tokens")
        if not isinstance(tokens, dict):
            return None

        try:
            # Support nested cache format: {cache: {read: N, write: N}} or flat {cacheRead: N}
            cache_obj = tokens.get("cache")
            if isinstance(cache_obj, dict):
                cache_r = int(cache_obj.get("read") or 0)
                cache_w = int(cache_obj.get("write") or 0)
            else:
                cache_r = int(tokens.get("cacheRead") or tokens.get("cache_read") or 0)
                cache_w = int(tokens.get("cacheWrite") or tokens.get("cache_write") or 0)
            result: dict[str, Any] = {
                "input": int(tokens.get("input") or 0),
                "output": int(tokens.get("output") or 0),
                "reasoning": int(tokens.get("reasoning") or 0),
                "cache_read": cache_r,
                "cache_write": cache_w,
                "total": int(tokens.get("total") or 0),
                "cost": float(tokens.get("cost") or 0.0),
            }
        except (TypeError, ValueError):
            return None

        # If total wasn't provided, compute it
        if result["total"] <= 0:
            result["total"] = result["input"] + result["output"] + result["reasoning"]

        # Check for compaction events in parts
        compaction = False
        summary = False
        parts = msg.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    if part.get("type") == "compaction":
                        compaction = True
                    if part.get("type") == "summary":
                        compaction = True
                        summary = True
        # Also check info.summary flag
        if info.get("summary"):
            summary = True

        result["compaction"] = compaction
        result["summary"] = summary
        return result

    def _print_context_status(self, turn: int, ctx_info: dict[str, Any], session_total: int) -> None:
        """Print [context] and [compact] status lines after each turn."""
        inp, out = ctx_info.get("input", 0), ctx_info.get("output", 0)
        ctx_len = self._context_length
        if ctx_len and ctx_len > 0:
            pct = inp / ctx_len * 100
            style = "bold yellow" if pct > 70 else "dim"
            _rprint(f"\\[context] turn {turn}: {inp:,}/{ctx_len:,} ({pct:.0f}%) | out={out:,} | session={session_total:,}", style)
        else:
            _rprint(f"\\[context] turn {turn}: in={inp:,} out={out:,} | session={session_total:,}", "dim")
        if ctx_info.get("compaction"):
            _rprint("\\[compact] auto-compaction triggered", "bold magenta")
        if ctx_info.get("summary"):
            _rprint("\\[compact] summary message (post-compaction)", "bold magenta")

    @staticmethod
    def _kill_popen(proc: subprocess.Popen | None, timeout: int = 5) -> None:
        """Terminate a Popen process (SIGTERM → wait → SIGKILL)."""
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            proc.wait(timeout=timeout)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _stop_local_server(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._request_json("POST", "/instance/dispose", body=None, require_auth=True, timeout_seconds=5)
            except Exception:
                pass
        self._kill_popen(self._proc)
        self._proc = None
        self._kill_popen(self._proxy_proc)
        self._proxy_proc = None
        for f in (self._server_log_file, self._proxy_log_file):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self._server_log_file = None
        self._proxy_log_file = None
        if self._temp_config_home is not None:
            shutil.rmtree(self._temp_config_home, ignore_errors=True)
            self._temp_config_home = None

    def run(self, text: str, *, on_turn=None) -> AgentResult:
        if self._tool_policy is not None:
            policy = replace(self._tool_policy, repo=self._repo.resolve(), unattended=self._unattended)
        else:
            policy = ToolPolicy(
                repo=self._repo.resolve(),
                unattended=self._unattended,
            )

        prompt = text
        trace: list[dict[str, Any]] = []
        self._stale_timeout_event = threading.Event()  # reset per run
        cumulative_tokens: dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
        for turn_idx in range(self._max_turns):
            monitor = None
            if on_turn:
                monitor = self._LLMWaitMonitor(
                    self._token_log_path, turn_idx + 1,
                    stale_timeout=self._stale_timeout,
                    server_proc=self._proc,
                )
                monitor._client_ref = self  # allow monitor to signal stale timeout
            try:
                if monitor:
                    monitor.start()
                try:
                    msg = self._post_message_with_retry(model=self._model_obj, text=prompt)
                except OpenCodeRequestError as e:
                    # Compatibility fallback: some builds may accept model as a string.
                    if e.status in (400, 422):
                        msg = self._post_message_with_retry(model=self._model_str, text=prompt)
                    else:
                        raise
            except StaleThinkingTimeout:
                raise  # propagate stale timeout without recovery
            finally:
                if monitor:
                    monitor.stop()

            # Extract and display context/token info
            ctx_info = self._extract_context_info(msg)
            if ctx_info is not None:
                cumulative_tokens["input"] += ctx_info["input"]
                cumulative_tokens["output"] += ctx_info["output"]
                cumulative_tokens["reasoning"] += ctx_info["reasoning"]
                session_total = cumulative_tokens["input"] + cumulative_tokens["output"] + cumulative_tokens["reasoning"]
                self._print_context_status(turn_idx + 1, ctx_info, session_total)

            opencode_err = None
            if isinstance(msg, dict):
                info = msg.get("info")
                if isinstance(info, dict):
                    err_obj = info.get("error")
                    if isinstance(err_obj, dict):
                        name = str(err_obj.get("name") or "").strip() or "Error"
                        data = err_obj.get("data")
                        detail = ""
                        if isinstance(data, dict):
                            detail = str(data.get("message") or "").strip()
                        if not detail:
                            detail = str(data).strip() if data is not None else ""
                        opencode_err = f"{name}: {detail}" if detail else name
            if opencode_err:
                raise RuntimeError(f"OpenCode agent error: {opencode_err}")

            if not isinstance(msg, dict):
                assistant_text = str(msg)
            else:
                parts = msg.get("parts")
                if not isinstance(parts, list):
                    assistant_text = str(msg)
                else:
                    texts: list[str] = []
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            t = part["text"]
                            if t.strip():
                                texts.append(t)
                    assistant_text = "\n".join(texts) or str(msg)
            calls = parse_tool_calls(assistant_text)
            turn_num = turn_idx + 1
            text_tail = tail(assistant_text or "", 4000)

            if not calls:
                entry: dict[str, Any] = {
                    "turn": turn_num, "assistant_text_tail": text_tail,
                    "calls": [], "results": [],
                }
                if ctx_info is not None:
                    entry["tokens"] = ctx_info
                trace.append(entry)
                if on_turn:
                    on_turn(TurnEvent(turn=turn_num, assistant_text=assistant_text, finished=True))
                return AgentResult(assistant_text=assistant_text, raw=msg, tool_trace=trace)

            # 逐个执行工具调用，每完成一个就回调 on_turn，实时显示进展
            results = execute_tool_calls(calls, policy=policy)
            compact_results: list[dict[str, Any]] = []
            for call_idx, r in enumerate(results):
                detail = dict(r.detail or {})
                for key in ("content", "stdout", "stderr"):
                    if isinstance(detail.get(key), str):
                        detail[key] = tail(detail[key], 4000)
                compact_results.append(detail | {"tool": r.kind, "ok": bool(r.ok)})

                # 实时回调：每完成一个工具调用就通知，让 stream printer 显示进展
                if on_turn:
                    on_turn(TurnEvent(
                        turn=turn_num,
                        assistant_text=assistant_text if call_idx == 0 else "",
                        calls=[calls[call_idx]],
                        results=[results[call_idx]],
                    ))

            entry = {
                "turn": turn_num, "assistant_text_tail": text_tail,
                "calls": [
                    {
                        "kind": str(c.kind),
                        "payload": c.payload if isinstance(c.payload, (dict, list, str, int, float, bool)) else str(c.payload),
                    }
                    for c in calls
                ],
                "results": compact_results,
            }
            if ctx_info is not None:
                entry["tokens"] = ctx_info
            trace.append(entry)

            prompt = format_tool_results(results)

        raise RuntimeError(f"OpenCode tool loop exceeded {self._max_turns} turns without a final response.")

    def _inject_permissions(self, env: dict) -> None:
        """Create temp opencode config with permission overrides."""
        xdg = str(env.get("XDG_CONFIG_HOME") or "").strip()
        cfg_home = Path(xdg) if xdg else Path.home() / ".config"
        cfg_path = cfg_home / "opencode" / "opencode.json"
        if not cfg_path.exists():
            return
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(cfg, dict):
            return
        perms = cfg.setdefault("permission", {})
        perms.update(self._permission_overrides)
        self._temp_config_home = Path(tempfile.mkdtemp(prefix="opencode_perm_"))
        tmp_dir = self._temp_config_home / "opencode"
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "opencode.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        env["XDG_CONFIG_HOME"] = str(self._temp_config_home)

    def _start_local_server(
        self,
        *,
        repo: Path,
        server_log_path: Path | None,
        username: str | None,
        append_log: bool = False,
    ) -> OpenCodeServerConfig:
        if not shutil.which("opencode"):
            raise RuntimeError("`opencode` not found in PATH. Install it from https://opencode.ai/")

        host = "127.0.0.1"
        port = _find_free_port(host)

        user = (username or "opencode").strip() or "opencode"
        pwd = secrets.token_urlsafe(24)
        env = dict(os.environ)
        # OpenCode's OpenAI-compatible provider reads `OPENAI_BASE_URL`.
        # Keep compatibility with `.env` files using `OPENAI_API_BASE`.
        if not str(env.get("OPENAI_BASE_URL") or "").strip():
            api_base = str(env.get("OPENAI_API_BASE") or "").strip().rstrip("/")
            if api_base:
                env["OPENAI_BASE_URL"] = api_base if api_base.endswith("/v1") else (api_base + "/v1")
        env["OPENCODE_SERVER_USERNAME"] = user
        env["OPENCODE_SERVER_PASSWORD"] = pwd
        if self._auto_compact is False:
            env["OPENCODE_DISABLE_AUTOCOMPACT"] = "true"

        # --- Start LLM streaming proxy ---
        if server_log_path is not None:
            self._start_llm_proxy(env, server_log_path.parent)

        # If proxy didn't create a temp config but we have permission overrides,
        # create one now to inject them.
        if self._permission_overrides and self._temp_config_home is None:
            self._inject_permissions(env)

        cmd = ["opencode", "serve", "--hostname", host, "--port", str(port)]

        stdout = subprocess.DEVNULL
        if server_log_path is not None:
            server_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
            self._server_log_file = server_log_path.open(
                "a" if append_log else "w",
                encoding="utf-8",
            )
            stdout = self._server_log_file

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stdout,
            start_new_session=True,
        )
        return OpenCodeServerConfig(base_url=f"http://{host}:{port}", username=user, password=pwd)

    def _start_llm_proxy(self, env: dict, log_dir: Path) -> None:
        """Start LLM streaming proxy and redirect OpenCode to use it.

        OpenCode reads its LLM API URL from ``~/.config/opencode/opencode.json``
        (via the provider's ``options.baseURL``), **not** from the
        ``OPENAI_BASE_URL`` environment variable.  So we must:

        1. Read the real ``baseURL`` from the OpenCode config.
        2. Start a local proxy pointing at that upstream.
        3. Create a temporary copy of the config with ``baseURL`` pointing
           at the proxy.
        4. Set ``XDG_CONFIG_HOME`` so the OpenCode server process picks up
           the temporary config.
        """
        proxy_script = Path(__file__).with_name("llm_proxy.py")
        if not proxy_script.exists():
            print("    [diag] WARNING: llm_proxy.py not found, token logging and stale timeout visualization disabled", flush=True)
            return

        # ---- 1. Find the real upstream URL ----
        upstream_url = ""
        config_data = None

        # Try OpenCode config (respects XDG_CONFIG_HOME)
        xdg = str(env.get("XDG_CONFIG_HOME") or "").strip()
        config_home = Path(xdg) if xdg else Path.home() / ".config"
        config_path = config_home / "opencode" / "opencode.json"
        provider_id = self._model_obj.get("providerID", "openai")

        if config_path.exists():
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                providers = config_data.get("provider", {})
                if provider_id in providers:
                    opts = providers[provider_id].get("options", {})
                    upstream_url = str(opts.get("baseURL", "")).strip().rstrip("/")
            except Exception:
                pass

        # Fallback: use OPENAI_BASE_URL env var
        if not upstream_url:
            upstream_url = str(env.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
            # Strip /v1 — the proxy forwards the full path including /v1
            if upstream_url.endswith("/v1"):
                upstream_url = upstream_url[:-3]

        if not upstream_url:
            print("    [diag] WARNING: no upstream URL found for LLM proxy, token logging disabled", flush=True)
            return

        # ---- 2. Start proxy process ----
        proxy_port = _find_free_port("127.0.0.1")

        log_dir.mkdir(parents=True, exist_ok=True)
        self._token_log_path = log_dir / "llm_tokens.log"
        self._token_log_path.write_text("", encoding="utf-8")

        proxy_log = log_dir / "llm_proxy.log"
        self._proxy_log_file = proxy_log.open("w", encoding="utf-8")

        proxy_timeout = int(self._stale_timeout + 300)
        self._proxy_proc = subprocess.Popen(
            [
                sys.executable, str(proxy_script),
                "--port", str(proxy_port),
                "--upstream", upstream_url,
                "--log", str(self._token_log_path),
                "--timeout", str(proxy_timeout),
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._proxy_log_file,
            stderr=self._proxy_log_file,
            text=True,
            start_new_session=True,
        )

        proxy_base = f"http://127.0.0.1:{proxy_port}"

        # ---- 3. Redirect OpenCode to the proxy ----
        # a) If we found an OpenCode config with the provider, create a temp
        #    copy with baseURL pointing at the proxy.
        if isinstance(config_data, dict) and provider_id in config_data.get("provider", {}):
            patched = copy.deepcopy(config_data)
            patched["provider"][provider_id].setdefault("options", {})["baseURL"] = proxy_base
            # Inject permission overrides (key is "permission" singular in OpenCode)
            if self._permission_overrides:
                patched.setdefault("permission", {}).update(self._permission_overrides)
            self._temp_config_home = Path(tempfile.mkdtemp(prefix="opencode_proxy_"))
            temp_cfg_dir = self._temp_config_home / "opencode"
            temp_cfg_dir.mkdir(parents=True)
            (temp_cfg_dir / "opencode.json").write_text(
                json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            env["XDG_CONFIG_HOME"] = str(self._temp_config_home)

        # b) Also set OPENAI_BASE_URL for providers that read it from env.
        env["OPENAI_BASE_URL"] = f"{proxy_base}/v1"

        # Wait for proxy to be ready (poll port instead of blind sleep)
        _proxy_ready = False
        for _pw in range(50):  # up to 10s (50 * 0.2s)
            try:
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.5):
                    _proxy_ready = True
                    break
            except OSError:
                time.sleep(0.2)
        if not _proxy_ready:
            print(f"    [diag] WARNING: LLM proxy on port {proxy_port} not ready after 10s, continuing without proxy", flush=True)
            # Kill the proxy process since it's not working
            try:
                self._proxy_proc.kill()
            except Exception:
                pass
            self._proxy_proc = None
            self._token_log_path = None
            return

    def _wait_health(self, label: str = "server") -> None:
        """Poll /global/health until the server responds (up to 60s)."""
        start = time.time()
        last_err = ""
        dots = 0
        while time.time() - start < 60:
            try:
                self._request_json("GET", "/global/health", body=None, require_auth=bool(self._server.password))
                print(f"    {label} ready ({time.time() - start:.1f}s)", flush=True)
                return
            except Exception as e:
                last_err = str(e)
                dots += 1
                if dots % 25 == 0:
                    print(f"    waiting for {label}... ({time.time() - start:.0f}s)", flush=True)
                time.sleep(0.2)
        raise RuntimeError(f"OpenCode {label} failed health check: {tail(last_err, 2000)}")

    def _create_session(self) -> None:
        """Create a new session and store the session ID."""
        data = self._request_json("POST", "/session", body={"title": self._session_title},
                                  require_auth=bool(self._server.password))
        sid = None
        if isinstance(data, dict):
            sid = data.get("id") or data.get("sessionID")
        if not isinstance(sid, str) or not sid.strip():
            raise RuntimeError(f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}")
        self._session_id = sid

    def _recover_server(self, reason: str, recover_idx: int) -> None:
        """Restart the local server and create a fresh session."""
        _exit_code = self._proc.poll() if self._proc else "no_proc"
        print(f"    [diag] transport error: {reason}", flush=True)
        print(f"    [diag] server exit_code={_exit_code}, recover_try={recover_idx}", flush=True)
        username = self._server.username if self._server else "opencode"
        self._stop_local_server()
        self._server = self._start_local_server(repo=self._repo, server_log_path=self._server_log_path,
                                                 username=username, append_log=True)
        self._wait_health("recovered server")
        self._create_session()

    @staticmethod
    def _is_transport_error(e: OpenCodeRequestError) -> bool:
        if e.status is not None:
            return False
        d = str(e.detail or "").lower()
        return any(n in d for n in (
            "connection refused", "connection reset", "connection aborted",
            "connection closed", "remote end closed", "timed out", "timeout",
            "network is unreachable", "name or service not known",
        ))

    def _post_message_with_retry(self, *, model: Any, text: str) -> Any:
        attempts = 1 + self._request_retry_attempts
        include_context = True
        last_err: OpenCodeRequestError | None = None
        recover_budget = self._session_recover_attempts if self._owns_local_server else 0
        recover_tries = 0

        for attempt in range(1, attempts + 1):
            try:
                return self._post_message(model=model, text=text, include_context=include_context)
            except OpenCodeRequestError as e:
                last_err = e
                # Degrade: some builds reject contextLength field
                if include_context and self._context_length and e.status in (400, 422):
                    include_context = False
                    try:
                        return self._post_message(model=model, text=text, include_context=False)
                    except OpenCodeRequestError as e2:
                        last_err = e2
                        e = e2

                # Stale timeout — propagate immediately
                if getattr(self, '_stale_timeout_event', None) and self._stale_timeout_event.is_set():
                    raise StaleThinkingTimeout("LLM stale thinking timeout") from e

                # Server recovery
                if self._is_transport_error(e) and recover_tries < recover_budget:
                    recover_tries += 1
                    try:
                        self._recover_server(e.detail, recover_tries)
                    except Exception as re:
                        last_err = OpenCodeRequestError(method=e.method, url=e.url, status=e.status,
                                                         detail=f"{e.detail}; recover_failed: {tail(str(re), 1200)}")
                    else:
                        delay = min(30.0, self._session_recover_backoff_seconds * (2 ** (recover_tries - 1)))
                        if delay > 0:
                            time.sleep(delay)
                        continue

                # Retry logic
                should_retry = e.status is None or (isinstance(e.status, int) and (e.status in (408, 409, 425, 429) or e.status >= 500))
                if attempt >= attempts or not should_retry:
                    raise
                delay = min(30.0, self._request_retry_backoff_seconds * (2 ** (attempt - 1)))
                if delay > 0:
                    time.sleep(delay)

        if last_err is not None:
            raise last_err
        raise RuntimeError("opencode_retry_failed_without_error")

    def _post_message(self, *, model: Any, text: str, include_context: bool = True) -> Any:
        s = str(text or "")
        cap = self._max_prompt_chars
        if cap is None or cap <= 0 or len(s) <= cap:
            clipped_text = s
        elif cap < 128:
            clipped_text = s[-cap:]
        else:
            marker = "\n...[TRUNCATED_FOR_OPENCODE_CONTEXT]...\n"
            head = max(32, cap // 2)
            tail_keep = max(32, cap - head - len(marker))
            clipped_text = s[:head] + marker + s[-tail_keep:]
        body = {
            "agent": "build",
            "model": model,
            "parts": [{"type": "text", "text": clipped_text}],
        }
        if include_context and self._context_length is not None:
            # Best-effort: different OpenCode versions may ignore this field.
            body["contextLength"] = int(self._context_length)
        # HTTP 超时自动跟随 stale_timeout：让 stale monitor 做真正的超时控制，
        # HTTP 层只是兜底安全网，永远不应先于 stale monitor 触发。
        msg_timeout = max(self._timeout_seconds, self._stale_timeout + 120)
        if not getattr(self, '_diag_timeout_printed', False):
            print(f"    [diag] timeout config: http={self._timeout_seconds}s, stale={self._stale_timeout}s, msg_deadline={msg_timeout}s", flush=True)
            self._diag_timeout_printed = True
        data = self._request_json(
            "POST",
            f"/session/{self._session_id}/message",
            body=body,
            require_auth=bool(self._server.password),
            timeout_seconds=msg_timeout,
        )
        # Some OpenCode builds/transports may respond with 200 + empty body; treat it as a transient transport failure
        # so the caller can retry or recover the local session instead of silently returning "None".
        if data is None:
            url = f"{self._server.base_url}/session/{self._session_id}/message"
            raise OpenCodeRequestError(method="POST", url=url, status=None, detail="connection closed: empty_response_body")
        return data

    def _request_json(self, method: str, path: str, *, body: Any, require_auth: bool, timeout_seconds: float | None = None) -> Any:
        url = f"{self._server.base_url}{path}"
        headers = {"Accept": "application/json"}
        if require_auth and self._server.password:
            token = base64.b64encode(
                f"{self._server.username}:{self._server.password}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = Request(url, method=method, data=data, headers=headers)
        timeout = self._timeout_seconds if timeout_seconds is None else float(timeout_seconds)

        # Use a background thread for the HTTP request so we can check
        # the stale timeout flag and abort early instead of blocking for
        # the full HTTP timeout (300s).
        result_container: list = []  # [value] or []
        error_container: list = []   # [exception] or []
        resp_ref: list = []          # [response] — 用于外部强制关闭

        def _do_request():
            try:
                resp = urlopen(req, timeout=timeout)
                resp_ref.append(resp)
                try:
                    raw = resp.read()
                    if not raw:
                        result_container.append(None)
                    else:
                        result_container.append(json.loads(raw.decode("utf-8", errors="replace")))
                finally:
                    try:
                        resp.close()
                    except Exception:
                        pass
            except Exception as exc:
                error_container.append(exc)

        req_thread = threading.Thread(target=_do_request, daemon=True)
        req_thread.start()

        # Poll every 2s, checking for stale timeout flag
        deadline = time.time() + timeout
        while time.time() < deadline:
            req_thread.join(timeout=2.0)
            if not req_thread.is_alive():
                break
            # Check if stale timeout was triggered — abort immediately
            if getattr(self, '_stale_timeout_event', None) and self._stale_timeout_event.is_set():
                # 强制关闭底层连接，让后台线程的 read() 立即失败退出
                for r in resp_ref:
                    try:
                        r.close()
                    except Exception:
                        pass
                # 等线程退出（server 已被 kill，socket 应该很快 RST）
                req_thread.join(timeout=5.0)
                raise StaleThinkingTimeout(
                    "LLM stale thinking timeout — aborting HTTP request"
                )

        if req_thread.is_alive():
            # Thread still running after deadline — force close and treat as timeout
            for r in resp_ref:
                try:
                    r.close()
                except Exception:
                    pass
            if "/message" in path:
                print(f"    [diag] HTTP polling deadline reached: timeout={timeout}s, url={path}", flush=True)
            raise OpenCodeRequestError(
                method=method, url=url, status=None,
                detail=f"timeout: request exceeded {timeout}s",
            )

        if error_container:
            e = error_container[0]
            if "/message" in path:
                print(f"    [diag] HTTP thread error on message POST: {type(e).__name__}: {e}", flush=True)
            if isinstance(e, HTTPError):
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(e)
                raise OpenCodeRequestError(method=method, url=url, status=int(getattr(e, "code", 0) or 0), detail=tail(detail, 2000))
            elif isinstance(e, URLError):
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=str(e))
            elif isinstance(e, (TimeoutError, socket.timeout)):
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"timeout: {e}")
            elif isinstance(e, OSError):
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"os_error: {e}")
            else:
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"error: {e}")

        return result_container[0] if result_container else None

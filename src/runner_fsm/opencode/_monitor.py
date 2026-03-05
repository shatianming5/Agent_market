"""LLM wait monitor: tails token log and detects stale thinking."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from ._types import _rprint


class LLMWaitMonitor:
    """Tail token log (from LLM proxy) + heartbeat while waiting for LLM response.

    Token log format (written by llm_proxy.py):
        [HH:MM:SS] >>> stream start model=glm-4.7
        <raw token content, no newlines between tokens>
        [HH:MM:SS] <<< stream end 523 tokens 4.2s

    Output format:
        | >>> stream start model=glm-4.7
        | 让我先看看数据格式，用 terminal 执行以下命令：...
        | <<< stream end 523 tokens 4.2s
    """

    def __init__(self, token_log: Path | None, turn: int,
                 heartbeat_interval: float = 5.0,
                 stale_timeout: float = 180.0,
                 server_proc=None):
        self._token_log = token_log
        self._turn = turn
        self._interval = heartbeat_interval
        self._stale_timeout = stale_timeout
        self._server_proc = server_proc
        self._client_ref = None  # set externally to flag stale timeout on client
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0
        self._log_offset = 0
        self._line_buf = ""
        self._proxy_output_tokens = 0
        self._proxy_rounds = 0
        self._proxy_stream_chars = 0

    def start(self):
        self._start = time.time()
        if self._token_log:
            try:
                self._log_offset = self._token_log.stat().st_size
            except Exception:
                self._log_offset = 0
            _rprint(f"... waiting for LLM response (turn {self._turn})", "dim italic")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                print("    [diag] WARNING: LLM monitor thread did not exit within 2s", flush=True)
        if self._line_buf.strip():
            text = self._line_buf.strip()
            if len(text) > 200:
                text = text[:197] + "..."
            self._print_token_line(text)
            self._line_buf = ""

    def _estimate_tokens(self) -> int:
        exact = self._proxy_output_tokens
        estimated = self._proxy_stream_chars // 4
        return max(exact, estimated)

    def _token_suffix(self) -> str:
        est = self._estimate_tokens()
        if est > 0:
            s = f" | ~{est:,} tokens"
            return s + f" / {self._proxy_rounds} rounds" if self._proxy_rounds > 1 else s
        if self._proxy_rounds > 0:
            return f" | round {self._proxy_rounds} streaming" if self._proxy_rounds > 1 else " | streaming"
        return ""

    def _print_thinking(self, elapsed: float):
        _rprint(f"... LLM thinking ({elapsed:.0f}s){self._token_suffix()}", "dim italic")

    def _print_tool_call(self, tool_name: str, elapsed: float):
        _rprint(f"... agent calling: {tool_name} ({elapsed:.0f}s){self._token_suffix()}", "bold cyan")

    def _print_tool_detail(self, detail: str):
        _rprint(detail[:200], "cyan")

    def _print_tool_content(self, text: str):
        text = text[:150]
        if text.startswith("+") and not text.startswith("+++"):
            _rprint(f"  {text}", "green")
        elif text.startswith("-") and not text.startswith("---"):
            _rprint(f"  {text}", "red")
        else:
            _rprint(f"  {text}", "dim")

    def _print_token_line(self, text: str):
        _rprint(f"| {text[:200]}", "dim")

    def _run(self):
        last_activity = time.time()
        last_real_activity = time.time()
        while not self._stop.wait(0.3):
            activity_type = self._tail_token_log()
            now = time.time()
            if activity_type:
                last_activity = now
                if activity_type == "content":
                    last_real_activity = now
            elif (now - last_activity) >= self._interval:
                elapsed = now - self._start
                stale = now - last_real_activity
                if self._stale_timeout and stale > self._stale_timeout:
                    _rprint(f"LLM stale for {stale:.0f}s (>{self._stale_timeout:.0f}s), aborting...", "bold yellow")
                    if self._client_ref is not None:
                        self._client_ref._stale_timeout_event.set()
                    if self._server_proc:
                        try:
                            self._server_proc.kill()
                        except Exception:
                            pass
                    break
                self._print_thinking(elapsed)
                last_activity = now

    def _tail_token_log(self) -> str | None:
        """Tail the token log. Returns "content" for real LLM output,
        "heartbeat" for proxy keep-alive signals, or None for no activity."""
        if not self._token_log:
            return None
        try:
            size = self._token_log.stat().st_size
            if size <= self._log_offset:
                return None
            with open(self._token_log, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_offset)
                new = f.read(size - self._log_offset)
            self._log_offset = size
            if not new:
                return None
        except Exception:
            return None

        has_content = False
        has_heartbeat = False
        for ch in new:
            if ch == "\n":
                line = self._line_buf.strip()
                self._line_buf = ""
                if not line:
                    continue
                if "<<< stream" in line:
                    try:
                        parts_s = line.split()
                        for i, w in enumerate(parts_s):
                            if w == "tokens" and i > 0:
                                self._proxy_output_tokens += int(parts_s[i - 1])
                                break
                    except (ValueError, IndexError):
                        pass
                    has_heartbeat = True
                    continue
                if ">>> stream" in line:
                    self._proxy_rounds += 1
                    has_heartbeat = True
                    continue
                if line.startswith("[HEARTBEAT]"):
                    has_heartbeat = True
                    continue
                if ("[DBG" in line
                        or '"encrypted_content"' in line
                        or ('"type":"response.' in line and len(line) > 150)):
                    continue
                if line.startswith("[TOOL_CONTENT] "):
                    content = line[15:]
                    self._print_tool_content(content)
                    has_content = True
                    continue
                if line.startswith("[TOOL_DETAIL] "):
                    detail = line[14:].strip()
                    self._print_tool_detail(detail)
                    has_content = True
                    continue
                if line.startswith("[TOOL] "):
                    tool_name = line[7:].strip()
                    elapsed = time.time() - self._start
                    self._print_tool_call(tool_name, elapsed)
                    has_content = True
                    continue
                self._proxy_stream_chars += len(line)
                self._print_token_line(line)
                has_content = True
            else:
                self._line_buf += ch
                buf_len = len(self._line_buf)
                if buf_len >= 80 and "\n" not in self._line_buf:
                    text = self._line_buf.strip()
                    if (text.startswith("[TOOL") or ">>> stream" in text or "<<< stream" in text) and buf_len < 500:
                        pass
                    else:
                        if text and "[DBG" not in text:
                            self._proxy_stream_chars += len(text)
                            self._print_token_line(text)
                            has_content = True
                        self._line_buf = ""
        if has_content:
            return "content"
        if has_heartbeat:
            return "heartbeat"
        return None

"""Transparent LLM API proxy that logs streaming tokens to a file.

Architecture:
    OpenCode Server  ---->  This Proxy  ---->  Real LLM API
                              |
                              v
                         token log file  ---->  _LLMWaitMonitor tails this

Usage:
    python llm_proxy.py --port 8201 --upstream https://real-api/v1 --log /tmp/tokens.log

The proxy:
1. Forwards all requests to the upstream LLM API unchanged
2. For streaming responses (SSE), intercepts chunks and writes decoded tokens to the log
3. For non-streaming responses, forwards as-is
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.request import Request, urlopen
from urllib.error import HTTPError


_upstream: str = ""
_log_path: str = ""
_debug: bool = False
_upstream_timeout: int = 600

# Persistent log file handle + lock (Fix 10: avoid open/close per write)
_log_lock = threading.Lock()
_log_file = None  # type: ignore


def _open_log():
    """Open the persistent log file handle. Called once from main()."""
    global _log_file
    if _log_path:
        _log_file = open(_log_path, "a", encoding="utf-8")


def _close_log():
    """Close the persistent log file handle."""
    global _log_file
    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None


def _write_log(msg: str):
    with _log_lock:
        try:
            if _log_file is not None:
                _log_file.write(msg)
                _log_file.flush()
        except Exception:
            pass


def _extract_token(chunk: dict) -> str:
    """Extract token content from an SSE chunk, trying multiple formats."""
    # Standard OpenAI format: choices[].delta.content
    for c in chunk.get("choices") or []:
        delta = c.get("delta") or {}
        content = delta.get("content")
        if content:
            return content
        # Some models put text in "text" instead of "content"
        text = delta.get("text")
        if text:
            return text
        # Reasoning/thinking content
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return reasoning

    # Non-standard: top-level content (string only; list is Anthropic-style below)
    top_content = chunk.get("content")
    if isinstance(top_content, str) and top_content:
        return top_content

    # Anthropic-style: content[].text (content is a list of blocks)
    if isinstance(top_content, list):
        for block in top_content:
            if isinstance(block, dict) and block.get("text"):
                return block["text"]

    return ""


def _log_tool_event(chunk: dict):
    """Extract tool call events from SSE chunks and write to token log."""
    event_type = chunk.get("type", "")

    # Responses API: added → tool name
    if event_type == "response.output_item.added":
        item = chunk.get("item") or {}
        if item.get("type") == "function_call":
            _write_log(f"\n[TOOL] {item.get('name', '?')}\n")
            return

    # Responses API: done → tool name + detail
    if event_type == "response.output_item.done":
        item = chunk.get("item") or {}
        if item.get("type") == "function_call":
            name = item.get("name", "?")
            detail = _extract_tool_detail(name, item.get("arguments", ""))
            if detail:
                _write_log(f"\n[TOOL_DETAIL] {name}: {detail}\n")
            return

    # Chat Completions API
    for c in chunk.get("choices") or []:
        for tc in (c.get("delta") or {}).get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if name:
                _write_log(f"\n[TOOL] {name}\n")
                return


def _extract_tool_detail(name: str, args_str: str) -> str:
    """Extract a concise summary from tool call arguments for logging."""
    if not args_str:
        return ""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return args_str[:120].replace("\n", " ").strip()

    if not isinstance(args, dict):
        return ""

    # Tool-specific key extraction
    if name == "bash":
        cmd = args.get("command", "")
        return cmd.replace("\n", " ; ").strip()[:200] if cmd else ""
    if name in ("read", "write", "edit"):
        return args.get("filePath", "") or args.get("file_path", "")
    if name in ("glob", "Glob"):
        return args.get("pattern", "")
    if name in ("grep", "Grep"):
        pat, path = args.get("pattern", ""), args.get("path", "")
        return f"{pat} in {path}" if pat and path else pat

    # Generic fallback: first short string value
    for v in args.values():
        if isinstance(v, str) and v.strip() and len(v) < 200:
            return v.strip()[:120]
    return ""


def _parse_sse_line(text: str) -> str | None:
    """Parse an SSE data line, return the payload or None if not a data line."""
    # Standard: "data: {...}" or "data: [DONE]"
    if text.startswith("data: "):
        return text[6:]
    # No-space variant: "data:{...}"
    if text.startswith("data:"):
        return text[5:]
    return None


def _strip_encrypted(obj: dict | list, _depth: int = 0) -> bool:
    """Recursively strip ``encrypted_content`` fields from a request payload.

    OpenAI Codex models return encrypted reasoning tokens tied to a specific
    organization.  When the conversation history is sent back through a
    different org / Azure deployment, the server rejects them with
    ``invalid_encrypted_content``.  Removing these fields allows multi-turn
    conversations to work across different endpoints.

    Returns True if any field was removed.
    """
    if _depth > 20:
        return False
    changed = False
    if isinstance(obj, dict):
        if "encrypted_content" in obj:
            del obj["encrypted_content"]
            changed = True
        for v in obj.values():
            if isinstance(v, (dict, list)):
                changed |= _strip_encrypted(v, _depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                changed |= _strip_encrypted(item, _depth + 1)
        # Remove reasoning items that are now empty after stripping
        before = len(obj)
        obj[:] = [
            item for item in obj
            if not (isinstance(item, dict) and item.get("type") == "reasoning"
                    and "encrypted_content" not in item
                    and not item.get("summary"))
        ]
        if len(obj) != before:
            changed = True
    return changed


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward requests to upstream, intercept streaming responses."""

    def log_message(self, format, *args):
        # Suppress default access log to stderr
        pass

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len else b""

        # Build upstream request
        url = f"{_upstream}{self.path}"
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        for key in ("Authorization", "Accept"):
            val = self.headers.get(key)
            if val:
                headers[key] = val

        # Check if request asks for streaming
        is_stream = False
        try:
            req_json = json.loads(body)
            is_stream = req_json.get("stream", False)
            model = req_json.get("model", "?")
            # Strip encrypted_content from requests to avoid cross-org
            # decryption failures in multi-turn conversations.
            if _strip_encrypted(req_json):
                body = json.dumps(req_json).encode("utf-8")
                headers["Content-Type"] = "application/json"
        except Exception:
            model = "?"

        req = Request(url, data=body, headers=headers, method="POST")

        try:
            resp = urlopen(req, timeout=_upstream_timeout)
        except HTTPError as e:
            try:
                self.send_response(e.code)
                for k, v in e.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(e.read())
            finally:
                e.close()
            return
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())
            return

        # Forward response headers
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in ("transfer-encoding", "connection"):
                self.send_header(k, v)
        self.end_headers()

        if not is_stream:
            try:
                data = resp.read()
                self.wfile.write(data)
            finally:
                resp.close()
            return

        # Streaming: read SSE chunks, log tokens, forward to client.
        # Use a reader thread + queue so heartbeats can be emitted even when
        # the upstream LLM is silent (e.g. during extended reasoning/thinking).
        _write_log(f"\n[{time.strftime('%H:%M:%S')}] >>> stream start model={model}\n")
        token_count = 0
        raw_lines = 0
        t0 = time.time()
        _last_heartbeat = t0

        _SENTINEL = None  # signals reader thread is done

        line_q: queue.Queue = queue.Queue(maxsize=256)

        def _reader():
            """Background thread: blocking readline() → queue."""
            try:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    line_q.put(line)
            except Exception as exc:
                line_q.put(exc)
            finally:
                line_q.put(_SENTINEL)
                try:
                    resp.close()
                except Exception:
                    pass

        reader_t = threading.Thread(target=_reader, daemon=True)
        reader_t.start()

        try:
            while True:
                try:
                    item = line_q.get(timeout=10.0)
                except queue.Empty:
                    # No data from upstream in 10s — emit heartbeat
                    _now = time.time()
                    _write_log(f"\n[HEARTBEAT] {raw_lines} chunks {_now - t0:.0f}s\n")
                    _last_heartbeat = _now
                    continue

                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item

                line = item
                # Forward raw bytes to OpenCode server
                self.wfile.write(line)
                self.wfile.flush()

                # Heartbeat on data arrival too (if interval elapsed)
                _now = time.time()
                if _now - _last_heartbeat >= 10.0:
                    _write_log(f"\n[HEARTBEAT] {raw_lines} chunks {_now - t0:.0f}s\n")
                    _last_heartbeat = _now

                # Parse SSE data lines for token content
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                payload = _parse_sse_line(text)
                if payload is None:
                    # Not an SSE data line — log for debugging on first few
                    if _debug and raw_lines < 3:
                        _write_log(f"[DBG non-data] {text[:200]}\n")
                    continue

                raw_lines += 1
                if payload.strip() == "[DONE]":
                    continue

                try:
                    chunk = json.loads(payload)
                    content = _extract_token(chunk)
                    if content:
                        token_count += 1
                        _write_log(content)
                    else:
                        # 提取 server 内部工具调用事件，写入 token log 供 monitor 展示进度
                        _log_tool_event(chunk)
                        if _debug and raw_lines <= 3:
                            _write_log(f"[DBG no-content] {payload[:300]}\n")
                except json.JSONDecodeError:
                    if _debug and raw_lines <= 3:
                        _write_log(f"[DBG bad-json] {payload[:300]}\n")
        except Exception as e:
            _write_log(f"\n[ERR] {e}\n")

        elapsed = time.time() - t0
        _write_log(f"\n[{time.strftime('%H:%M:%S')}] <<< stream end {token_count} tokens {elapsed:.1f}s (data_lines={raw_lines})\n")

    def do_GET(self):
        # Forward GET requests (e.g. /v1/models)
        url = f"{_upstream}{self.path}"
        headers = {}
        for key in ("Authorization", "Accept"):
            val = self.headers.get(key)
            if val:
                headers[key] = val
        req = Request(url, headers=headers, method="GET")
        try:
            resp = urlopen(req, timeout=30)
            try:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
            finally:
                resp.close()
        except HTTPError as e:
            try:
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(e.read())
            finally:
                e.close()
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--debug", action="store_true", help="Log raw SSE data for debugging")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Upstream request timeout in seconds (default: 600)")
    args = parser.parse_args()

    global _upstream, _log_path, _debug, _upstream_timeout
    _upstream = args.upstream.rstrip("/")
    _log_path = args.log
    _debug = args.debug
    _upstream_timeout = max(30, args.timeout)

    _open_log()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
        print(f"LLM proxy listening on 127.0.0.1:{args.port} -> {_upstream} (timeout={_upstream_timeout}s)", flush=True)
        server.serve_forever()
    finally:
        _close_log()


if __name__ == "__main__":
    main()

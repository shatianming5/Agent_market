"""Shared types and small helper functions for the OpenCode client."""
from __future__ import annotations

import atexit
import socket
from dataclasses import dataclass
from typing import Any


# Module-level tracking of all active OpenCodeClient instances for cleanup on signal
_active_clients: set = set()


def _cleanup_active_clients() -> None:
    """Best-effort cleanup of all tracked OpenCodeClient instances."""
    for client in list(_active_clients):
        try:
            client.close()
        except Exception:
            pass


atexit.register(_cleanup_active_clients)


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _rprint(text: str, style: str = "dim") -> None:
    """Print with rich formatting if available, else plain."""
    try:
        from pipeline.ui import console
        console.print(f"    [{style}]{text}[/]")
    except Exception:
        print(f"    {text}", flush=True)


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Find a free port with SO_REUSEADDR to reduce TOCTOU race window."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        _, port = s.getsockname()
    return port


@dataclass(frozen=True)
class OpenCodeServerConfig:
    base_url: str
    username: str
    password: str


class OpenCodeRequestError(RuntimeError):
    def __init__(self, *, method: str, url: str, status: int | None, detail: str):
        super().__init__(f"OpenCode request failed: {method} {url} ({status}) {detail}")
        self.method = method
        self.url = url
        self.status = status
        self.detail = detail


class StaleThinkingTimeout(RuntimeError):
    """Raised when the LLM has been thinking without progress for too long."""
    pass

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="am-server",
        description="Run the Agent Market FastAPI service via uvicorn.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", default=8032, type=int, help="Bind port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Log level passed to uvicorn (debug, info, warning, error).",
    )
    parser.add_argument(
        "--proxy-headers",
        action="store_true",
        help="Enable processing of X-Forwarded-* headers.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        proxy_headers=args.proxy_headers,
    )


if __name__ == "__main__":
    main()

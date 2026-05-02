#!/usr/bin/env python3
"""Convert a VLESS subscription into a local Xray client config.

This utility intentionally supports the subset used by the server-side Docker
proxy workflow: VLESS outbounds with tcp/ws transport and none/tls/reality
security.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


def _read_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    raw_lines = [line.strip() for line in text.splitlines() if "://" in line]
    if raw_lines:
        return raw_lines

    compact = re.sub(r"\s+", "", text)
    decoded = base64.b64decode(compact + "===", validate=False).decode("utf-8", errors="ignore")
    return [line.strip() for line in decoded.splitlines() if "://" in line]


def _query(parsed) -> dict[str, str]:
    return {key: values[-1] for key, values in parse_qs(parsed.query, keep_blank_values=True).items() if values}


def _node_name(parsed) -> str:
    return unquote(parsed.fragment or "").strip()


def _stream_settings(parsed, q: dict[str, str]) -> dict[str, Any]:
    network = q.get("type") or q.get("network") or "tcp"
    security = q.get("security") or "none"
    stream: dict[str, Any] = {"network": network, "security": security}

    if security == "tls":
        tls: dict[str, Any] = {}
        server_name = q.get("sni") or q.get("serverName") or q.get("host") or parsed.hostname
        if server_name:
            tls["serverName"] = server_name
        if q.get("fp"):
            tls["fingerprint"] = q["fp"]
        if q.get("alpn"):
            tls["alpn"] = [part for part in q["alpn"].split(",") if part]
        if q.get("allowInsecure") in {"1", "true", "True"}:
            tls["allowInsecure"] = True
        stream["tlsSettings"] = tls
    elif security == "reality":
        reality: dict[str, Any] = {}
        server_name = q.get("sni") or q.get("serverName")
        if server_name:
            reality["serverName"] = server_name
        if q.get("fp"):
            reality["fingerprint"] = q["fp"]
        if q.get("pbk"):
            reality["publicKey"] = q["pbk"]
        if q.get("sid"):
            reality["shortId"] = q["sid"]
        if q.get("spx"):
            reality["spiderX"] = q["spx"]
        stream["realitySettings"] = reality

    if network == "ws":
        ws: dict[str, Any] = {}
        if q.get("path"):
            ws["path"] = q["path"]
        headers: dict[str, str] = {}
        if q.get("host"):
            headers["Host"] = q["host"]
        if headers:
            ws["headers"] = headers
        stream["wsSettings"] = ws
    elif network == "grpc":
        grpc: dict[str, Any] = {}
        if q.get("serviceName"):
            grpc["serviceName"] = q["serviceName"]
        stream["grpcSettings"] = grpc
    elif network == "tcp" and q.get("headerType") == "http":
        stream["tcpSettings"] = {"header": {"type": "http"}}

    return stream


def _vless_outbound(link: str) -> dict[str, Any]:
    parsed = urlparse(link)
    if parsed.scheme != "vless":
        raise ValueError(f"unsupported scheme: {parsed.scheme}")
    q = _query(parsed)
    user: dict[str, Any] = {
        "id": unquote(parsed.username or ""),
        "encryption": q.get("encryption") or "none",
    }
    if q.get("flow"):
        user["flow"] = q["flow"]
    return {
        "tag": "proxy",
        "protocol": "vless",
        "settings": {
            "vnext": [
                {
                    "address": parsed.hostname,
                    "port": int(parsed.port or 443),
                    "users": [user],
                }
            ]
        },
        "streamSettings": _stream_settings(parsed, q),
    }


def build_config(link: str, *, socks_port: int, http_port: int) -> dict[str, Any]:
    outbound = _vless_outbound(link)
    return {
        "log": {"loglevel": "warning"},
        "inbounds": [
            {
                "tag": "socks-in",
                "listen": "127.0.0.1",
                "port": socks_port,
                "protocol": "socks",
                "settings": {"udp": True},
            },
            {
                "tag": "http-in",
                "listen": "127.0.0.1",
                "port": http_port,
                "protocol": "http",
            },
        ],
        "outbounds": [
            outbound,
            {"tag": "direct", "protocol": "freedom"},
            {"tag": "block", "protocol": "blackhole"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-file", required=True)
    parser.add_argument("--output")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--socks-port", type=int, default=10808)
    parser.add_argument("--http-port", type=int, default=10809)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    links = [link for link in _read_links(Path(args.subscription_file)) if link.startswith("vless://")]
    if args.list:
        for idx, link in enumerate(links):
            parsed = urlparse(link)
            q = _query(parsed)
            print(
                json.dumps(
                    {
                        "index": idx,
                        "name": _node_name(parsed),
                        "security": q.get("security") or "none",
                        "network": q.get("type") or q.get("network") or "tcp",
                        "port": int(parsed.port or 443),
                    },
                    ensure_ascii=False,
                )
            )
        return

    if not links:
        raise SystemExit("no vless links found")
    if args.index < 0 or args.index >= len(links):
        raise SystemExit(f"index out of range: {args.index}; available={len(links)}")
    config = build_config(links[args.index], socks_port=args.socks_port, http_port=args.http_port)
    payload = json.dumps(config, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

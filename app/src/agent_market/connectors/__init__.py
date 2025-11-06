from __future__ import annotations

import logging
import shutil
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import httpx

try:
    from web3 import Web3  # type: ignore
except Exception:  # pragma: no cover
    Web3 = None  # type: ignore

logger = logging.getLogger(__name__)

HTTP_CLIENT_FACTORY: Callable[[float], httpx.Client] = lambda timeout: httpx.Client(timeout=timeout)


class ConnectorError(Exception):
    """Raised when a connector fails during credential validation or connectivity tests."""


class Connector(ABC):
    connector_type: str = "base"
    required_credentials: tuple[str, ...] = ()

    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]) -> None:
        self.config = config or {}
        self.credentials = credentials or {}

    def validate(self) -> None:
        missing = [field for field in self.required_credentials if field not in self.credentials or not self.credentials[field]]
        if missing:
            raise ConnectorError(f"missing credentials: {', '.join(missing)}")
        self._validate()

    def test_connection(self) -> Dict[str, Any]:
        self.validate()
        return self._test()

    def _validate(self) -> None:  # pragma: no cover - optional hooks
        """Hook for subclasses."""

    @abstractmethod
    def _test(self) -> Dict[str, Any]:
        """Perform a lightweight connectivity check returning metadata."""


class HTTPConnector(Connector):
    """Base class for connectors that interact over HTTP/HTTPS."""

    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]) -> None:
        super().__init__(config, credentials)
        self._last_call = 0.0

    @property
    def _timeout(self) -> float:
        return float(self.config.get("timeout", 5.0))

    @property
    def _retries(self) -> int:
        return max(1, int(self.config.get("retries", 3)))

    @property
    def _retry_delay(self) -> float:
        return float(self.config.get("retry_delay", 0.5))

    def _throttle(self) -> None:
        minimum = float(self.config.get("min_interval", 0.0))
        if minimum <= 0:
            return
        delta = time.monotonic() - self._last_call
        if delta < minimum:
            time.sleep(minimum - delta)

    def _request(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        self._throttle()
        last_exc: Optional[Exception] = None
        timeout = self._timeout
        for attempt in range(1, self._retries + 1):
            try:
                with HTTP_CLIENT_FACTORY(timeout) as client:
                    response = client.request(method, url, headers=headers, params=params)
                if response.status_code >= 400:
                    raise ConnectorError(f"HTTP {response.status_code} from {url}")
                self._last_call = time.monotonic()
                return response
            except Exception as exc:  # pragma: no cover - network errors
                last_exc = exc
                if attempt >= self._retries:
                    raise ConnectorError(f"request to {url} failed after {self._retries} attempts: {exc}") from exc
                time.sleep(self._retry_delay * attempt)
        raise ConnectorError(f"request to {url} failed: {last_exc}")

    def _get_json(self, url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._request("GET", url, headers=headers, params=params)
        try:
            return response.json()
        except Exception as exc:
            raise ConnectorError(f"invalid JSON response from {url}: {exc}") from exc


class ConnectorRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Connector]] = {}

    def register(self, connector_type: str, factory: Callable[[Dict[str, Any], Dict[str, Any]], Connector]) -> None:
        self._factories[connector_type] = factory

    def create(self, connector_type: str, config: Dict[str, Any], credentials: Dict[str, Any]) -> Connector:
        if connector_type not in self._factories:
            raise ConnectorError(f"unknown connector type '{connector_type}'")
        return self._factories[connector_type](config, credentials)

    def list_types(self) -> list[str]:
        return sorted(self._factories.keys())


REGISTRY = ConnectorRegistry()


class EVMConnector(Connector):
    connector_type = "evm"
    required_credentials = ("rpc_url",)

    def _validate(self) -> None:
        url = self.credentials.get("rpc_url")
        if not isinstance(url, str) or not url.startswith("http"):
            raise ConnectorError("rpc_url must be a valid HTTP(s) endpoint")

    def _test(self) -> Dict[str, Any]:
        url = self.credentials["rpc_url"]
        if Web3 is None:
            return {"success": True, "message": "web3 not installed; skipping live RPC check"}
        try:
            provider = Web3.HTTPProvider(url, request_kwargs={"timeout": self.config.get("timeout", 5)})
            web3 = Web3(provider)
            block = web3.eth.block_number  # type: ignore[attr-defined]
            return {"success": True, "latest_block": int(block)}
        except Exception as exc:
            raise ConnectorError(f"failed to query RPC: {exc}") from exc


class BinanceConnector(HTTPConnector):
    connector_type = "binance"
    required_credentials = ("api_key", "api_secret")

    def _test(self) -> Dict[str, Any]:
        base_url = self.config.get("base_url", "https://api.binance.com")
        url = f"{base_url.rstrip('/')}/api/v3/time"
        headers = {"X-MBX-APIKEY": self.credentials.get("api_key", "")}
        data = self._get_json(url, headers=headers)
        return {"success": True, "serverTime": data.get("serverTime")}


class OKXConnector(HTTPConnector):
    connector_type = "okx"
    required_credentials = ("api_key", "api_secret")

    def _test(self) -> Dict[str, Any]:
        base_url = self.config.get("base_url", "https://www.okx.com")
        url = f"{base_url.rstrip('/')}/api/v5/public/time"
        data = self._get_json(url)
        return {"success": True, "ts": data.get("data", [{}])[0].get("ts") if isinstance(data.get("data"), list) else None}


class CoinGeckoConnector(HTTPConnector):
    connector_type = "coingecko"
    required_credentials: tuple[str, ...] = ()

    def _test(self) -> Dict[str, Any]:
        base_url = self.config.get("base_url", "https://api.coingecko.com")
        url = f"{base_url.rstrip('/')}/api/v3/ping"
        data = self._get_json(url)
        return {"success": True, "reply": data.get("gecko_says")}


class CovalentConnector(HTTPConnector):
    connector_type = "covalent"
    required_credentials = ("api_key",)

    def _test(self) -> Dict[str, Any]:
        base_url = self.config.get("base_url", "https://api.covalenthq.com")
        url = f"{base_url.rstrip('/')}/v1/chains/status/"
        params = {"key": self.credentials["api_key"]}
        data = self._get_json(url, params=params)
        return {"success": True, "chains": len(data.get("data", [])) if isinstance(data.get("data"), list) else None}


class TelegramConnector(HTTPConnector):
    connector_type = "telegram"
    required_credentials = ("bot_token",)

    def _test(self) -> Dict[str, Any]:
        token = self.credentials["bot_token"]
        base_url = self.config.get("base_url", "https://api.telegram.org")
        url = f"{base_url.rstrip('/')}/bot{token}/getMe"
        data = self._get_json(url)
        if not data.get("ok"):
            raise ConnectorError("telegram API response not ok")
        return {"success": True, "username": data.get("result", {}).get("username")}


class DiscordConnector(HTTPConnector):
    connector_type = "discord"
    required_credentials = ("bot_token",)

    def _test(self) -> Dict[str, Any]:
        token = self.credentials["bot_token"]
        base_url = self.config.get("base_url", "https://discord.com")
        url = f"{base_url.rstrip('/')}/api/v10/users/@me"
        headers = {"Authorization": token if token.startswith("Bot ") else f"Bot {token}"}
        data = self._get_json(url, headers=headers)
        return {"success": True, "id": data.get("id"), "username": data.get("username")}


class MCPBrowserConnector(Connector):
    """Lightweight wrapper for MCP-compatible browser agents launched via local commands."""

    connector_type = "mcp_browser"
    required_credentials: tuple[str, ...] = ()

    def _validate(self) -> None:
        command = self.config.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ConnectorError("mcp_browser connector requires a non-empty 'command' in config")
        args = self.config.get("args")
        if args is not None:
            if not isinstance(args, (list, tuple)):
                raise ConnectorError("'args' must be a list when provided")
            if not all(isinstance(item, (str, int, float)) for item in args):
                raise ConnectorError("each entry in 'args' must be string/number")
        env = self.config.get("env")
        if env is not None and not isinstance(env, dict):
            raise ConnectorError("'env' must be a mapping when provided")
        cwd = self.config.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ConnectorError("'cwd' must be a string path when provided")

    def _test(self) -> Dict[str, Any]:
        command = self.config["command"]
        resolved = shutil.which(command)
        if not resolved:
            raise ConnectorError(f"command '{command}' not found on PATH")
        args = self.config.get("args") or []
        payload = {
            "success": True,
            "command": command,
            "resolved_path": resolved,
            "args": list(args),
        }
        env = self.config.get("env")
        if env:
            payload["env_keys"] = sorted(str(k) for k in env.keys())
        cwd = self.config.get("cwd")
        if cwd:
            payload["cwd"] = cwd
        return payload


def register_defaults() -> None:
    REGISTRY.register(EVMConnector.connector_type, lambda cfg, cred: EVMConnector(cfg, cred))
    REGISTRY.register(BinanceConnector.connector_type, lambda cfg, cred: BinanceConnector(cfg, cred))
    REGISTRY.register(OKXConnector.connector_type, lambda cfg, cred: OKXConnector(cfg, cred))
    REGISTRY.register(CoinGeckoConnector.connector_type, lambda cfg, cred: CoinGeckoConnector(cfg, cred))
    REGISTRY.register(CovalentConnector.connector_type, lambda cfg, cred: CovalentConnector(cfg, cred))
    REGISTRY.register(TelegramConnector.connector_type, lambda cfg, cred: TelegramConnector(cfg, cred))
    REGISTRY.register(DiscordConnector.connector_type, lambda cfg, cred: DiscordConnector(cfg, cred))
    REGISTRY.register(MCPBrowserConnector.connector_type, lambda cfg, cred: MCPBrowserConnector(cfg, cred))


register_defaults()

__all__ = [
    "Connector",
    "ConnectorError",
    "ConnectorRegistry",
    "HTTPConnector",
    "REGISTRY",
    "register_defaults",
    "HTTP_CLIENT_FACTORY",
    "MCPBrowserConnector",
]

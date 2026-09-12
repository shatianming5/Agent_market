#!/usr/bin/env python
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger("agent_market.freqtrade_cli")


_OFFLINE_MARKET_RUNMODE_TOKENS = ("backtest", "hyperopt", "edge", "lookahead", "recursive")
_OFFLINE_MARKET_COMMANDS = {"backtesting", "hyperopt", "edge", "lookahead-analysis", "recursive-analysis"}


def _bootstrap_freqtrade_sys_path() -> None:
    """Make freqtrade imports deterministic under repo executions.

    Some entrypoints run this wrapper with an inherited `PYTHONPATH=src:.`,
    which injects the repo root into `sys.path` and causes the top-level
    `freqtrade/` directory to be imported as a namespace package.

    Prefer the vendored package in `REPO_ROOT/freqtrade/freqtrade`.
    """
    repo_root = Path(__file__).resolve().parents[1]
    vendored_root = repo_root / "freqtrade"
    vendored_pkg_init = vendored_root / "freqtrade" / "__init__.py"
    if not vendored_pkg_init.exists():
        return

    try:
        repo_root_resolved = repo_root.resolve()
    except Exception:
        repo_root_resolved = repo_root

    sanitized: list[str] = []
    for entry in list(sys.path):
        try:
            resolved = Path.cwd().resolve() if not entry else Path(entry).resolve()
        except Exception:
            sanitized.append(entry)
            continue
        if resolved == repo_root_resolved:
            continue
        sanitized.append(entry)

    vendored_root_s = str(vendored_root)
    sanitized = [p for p in sanitized if p != vendored_root_s]
    sys.path[:] = [vendored_root_s] + sanitized


def _runmode_text(runmode: Any) -> str:
    value = getattr(runmode, "value", runmode)
    return str(value or "").lower()


def _should_synthesize_offline_markets(runmode: Any, optimize_modes: Any) -> bool:
    try:
        if runmode in optimize_modes:
            return True
    except Exception:
        pass
    text = _runmode_text(runmode)
    return any(token in text for token in _OFFLINE_MARKET_RUNMODE_TOKENS)


def _force_offline_markets_for_args(args: list[str]) -> bool:
    return any(arg in _OFFLINE_MARKET_COMMANDS for arg in args)


def _patch_offline_markets(*, force: bool = False) -> None:
    """Monkeypatch freqtrade to avoid exchange API calls in optimize modes.

    In backtesting/hyperopt modes, freqtrade often accesses `Exchange.markets`,
    which triggers `load_markets` (network). For an offline-reproducible pipeline,
    we synthesize minimal market metadata from configured pairs instead.
    """
    try:
        from freqtrade.enums import OPTIMIZE_MODES  # type: ignore
        from freqtrade.exchange.exchange import Exchange  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("freqtrade not available; cannot patch offline markets (%s)", exc)
        return

    prop = getattr(Exchange, "markets", None)
    original_fget: Optional[Callable[..., Any]] = getattr(prop, "fget", None)
    if original_fget is None:  # pragma: no cover
        logger.warning("freqtrade Exchange.markets is not a property; cannot patch")
        return

    def _patched_markets(self) -> Any:  # noqa: ANN001
        try:
            if not getattr(self, "_markets", None):
                cfg = getattr(self, "_config", None) or {}
                if force or _should_synthesize_offline_markets(cfg.get("runmode"), OPTIMIZE_MODES):
                    pairs = (
                        cfg.get("pairs")
                        or (cfg.get("exchange") or {}).get("pair_whitelist")
                        or []
                    )
                    synthesized: dict[str, Any] = {}
                    for pair in pairs:
                        if not isinstance(pair, str) or "/" not in pair:
                            continue
                        base, quote = pair.split("/", 1)
                        is_swap = ":" in quote
                        settle = None
                        if is_swap:
                            quote, settle = quote.split(":", 1)
                        synthesized[pair] = {
                            "symbol": pair,
                            "base": base,
                            "quote": quote,
                            "settle": settle,
                            "spot": not is_swap,
                            "margin": False,
                            "swap": is_swap,
                            "future": False,
                            "type": "swap" if is_swap else "spot",
                            "linear": is_swap,
                            "inverse": False,
                            "contract": is_swap,
                            "active": True,
                            "limits": {
                                "amount": {"min": None, "max": None},
                                "cost": {"min": None, "max": None},
                                "leverage": {"min": None, "max": None},
                            },
                            "precision": {"price": None, "amount": None},
                        }
                    if synthesized:
                        try:
                            logging.getLogger("freqtrade").warning(
                                "Markets not loaded (offline analysis mode). Using synthesized markets for offline run."
                            )
                        except Exception:
                            pass
                        setattr(self, "_markets", synthesized)
                        return synthesized
        except Exception:
            pass
        return original_fget(self)

    Exchange.markets = property(_patched_markets)  # type: ignore[assignment]


def _arg_values(args: list[str], flag: str) -> list[str]:
    values: list[str] = []
    prefix = f"{flag}="
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            values.append(args[i + 1])
        elif arg.startswith(prefix):
            values.append(arg[len(prefix):])
    return values


def _resolve_cli_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _config_preflight_paths(config_paths: list[str]) -> tuple[Optional[Path], list[Path]]:
    userdir: Optional[Path] = None
    datadirs: list[Path] = []
    for raw in config_paths:
        cfg_path = _resolve_cli_path(raw)
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if userdir is None and isinstance(payload.get("user_data_dir"), str) and payload["user_data_dir"].strip():
            userdir = _resolve_cli_path(payload["user_data_dir"])
        raw_datadir = payload.get("datadir")
        if isinstance(raw_datadir, str) and raw_datadir.strip():
            datadir = _resolve_cli_path(raw_datadir)
            if datadir not in datadirs:
                datadirs.append(datadir)
    return userdir, datadirs


def _preflight_raw_ohlcv(args: list[str]) -> None:
    """Thin shim: route to ``agent_market.freqtrade_preflight.assert_raw_ohlcv``.

    The shared helper is importable from any code path that invokes
    freqtrade (factor_lab/backtest.py, strategy_miner/_evaluation.py, etc.)
    so the preflight cannot be bypassed by going around this CLI wrapper.

    Codex review R3: also parses ``--datadir`` overrides so ``freqtrade
    backtesting --datadir <X>`` checks ``<X>`` rather than only the
    default ``user_data/data/``.
    """
    cmds = {"backtesting", "hyperopt", "trade", "edge", "lookahead-analysis", "recursive-analysis"}
    if not any(a in cmds for a in args):
        return
    userdir: Optional[Path] = None
    datadirs: list[Path] = []
    userdir_values = _arg_values(args, "--userdir")
    datadir_values = _arg_values(args, "--datadir")
    config_userdir, config_datadirs = _config_preflight_paths(_arg_values(args, "--config"))
    if userdir_values:
        userdir = _resolve_cli_path(userdir_values[-1])
    elif config_userdir is not None:
        userdir = config_userdir
    for raw in [*config_datadirs, *(_resolve_cli_path(value) for value in datadir_values)]:
        if raw not in datadirs:
            datadirs.append(raw)
    if userdir is None:
        repo_root = Path(__file__).resolve().parents[1]
        userdir = repo_root / "user_data"
    # Lazy import — keeps cold-start cheap when preflight is skipped.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from agent_market.freqtrade_preflight import assert_raw_ohlcv
    assert_raw_ohlcv(
        userdir,
        extra_datadirs=datadirs or None,
        include_userdir_data=not bool(datadirs),
    )


def main(argv: Optional[list[str]] = None) -> int:
    _bootstrap_freqtrade_sys_path()
    args = list(sys.argv[1:] if argv is None else argv)
    disable = False
    if args and args[0] == "--no-offline-markets":
        disable = True
        args = args[1:]
    skip_preflight = False
    if "--no-ohlcv-preflight" in args:
        skip_preflight = True
        args = [a for a in args if a != "--no-ohlcv-preflight"]
    if not disable:
        _patch_offline_markets(force=_force_offline_markets_for_args(args))
    if not skip_preflight:
        _preflight_raw_ohlcv(args)

    try:
        from freqtrade.main import main as freqtrade_main  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(
            "[agent_market] freqtrade is required for backtest/hyperopt.",
            file=sys.stderr,
        )
        print(
            "[agent_market] Install with: pip install -r requirements-full.txt",
            file=sys.stderr,
        )
        logger.error("freqtrade import failed: %s", exc)
        return 2

    freqtrade_main(args)
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())

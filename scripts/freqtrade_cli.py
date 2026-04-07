#!/usr/bin/env python
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger("agent_market.freqtrade_cli")


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


def _patch_offline_markets() -> None:
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
                if cfg.get("runmode") in OPTIMIZE_MODES:
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
                        synthesized[pair] = {
                            "symbol": pair,
                            "base": base,
                            "quote": quote,
                            "spot": True,
                            "margin": False,
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
                                "Markets not loaded (optimize mode). Using synthesized markets for offline run."
                            )
                        except Exception:
                            pass
                        setattr(self, "_markets", synthesized)
                        return synthesized
        except Exception:
            pass
        return original_fget(self)

    Exchange.markets = property(_patched_markets)  # type: ignore[assignment]


def main(argv: Optional[list[str]] = None) -> int:
    _bootstrap_freqtrade_sys_path()
    args = list(sys.argv[1:] if argv is None else argv)
    disable = False
    if args and args[0] == "--no-offline-markets":
        disable = True
        args = args[1:]
    if not disable:
        _patch_offline_markets()

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

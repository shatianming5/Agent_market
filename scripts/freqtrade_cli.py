#!/usr/bin/env python
from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Optional


logger = logging.getLogger("agent_market.freqtrade_cli")


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

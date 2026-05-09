"""Shared raw-OHLCV preflight for any code path that invokes freqtrade.

Codex review (runtime loop): `factor_lab features mtf|xs|...` writes
feature-augmented columns back into the raw OHLCV feather files, which
silently breaks freqtrade's feather datahandler with a `Length mismatch:
Expected axis has N elements, new values have 6 elements` error deep
inside `_ohlcv_load`. This module exposes a single guard function that
ANY code path invoking freqtrade should call BEFORE the subprocess /
import — it fail-closes with a clear remediation pointer rather than
letting freqtrade crash mid-load.

Usage::

    from agent_market.freqtrade_preflight import assert_raw_ohlcv

    assert_raw_ohlcv(userdir)   # raises SystemExit on contamination
    subprocess.run(["freqtrade", "backtesting", "--userdir", userdir, ...])

The check is OFF when env var ``AGENT_MARKET_NO_OHLCV_PREFLIGHT=1`` is
set, mirroring the ``--no-ohlcv-preflight`` CLI escape hatch in
``scripts/freqtrade_cli.py``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional


_OHLCV: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")
# Only check exchange-rooted OHLCV feathers: data/<exchange>/<PAIR>-<tf>.feather
# (or data/<exchange>/futures/<PAIR>-<tf>-futures.feather for OKX-style futures).
# Funding-rate / microstructure / cross-section feathers under sub-namespaces
# (data/funding/, data/micro/, ...) have different schemas — must NOT trigger.
_TF_GROUP = r"(1m|3m|5m|15m|30m|1h|2h|4h|6h|8h|12h|1d|3d|1w|1M)"
_OHLCV_TF_RE = re.compile(rf"^[A-Z]+_[A-Z]+-{_TF_GROUP}\.feather$")
_OHLCV_FUT_RE = re.compile(rf"^[A-Z]+_[A-Z]+-{_TF_GROUP}-futures\.feather$")


class OHLCVPreflightError(SystemExit):
    """Raised when raw OHLCV cache is contaminated. Inherits SystemExit so
    a top-level CLI exits with a non-zero status without a noisy
    traceback — matches the existing freqtrade_cli wrapper behavior."""


def _candidate_data_roots(
    userdir: Path,
    extra_datadirs: Optional[Iterable[Path | str]] = None,
) -> list[Path]:
    """Yield all directories that could hold raw OHLCV feathers.

    Codex review R3: `--datadir` overrides aren't picked up if we only
    look at ``userdir/data/``. Combine: (a) ``userdir/data`` (default
    freqtrade datadir layout) + (b) any explicit ``--datadir`` paths the
    caller supplies.
    """
    roots: list[Path] = []
    primary = Path(userdir) / "data"
    if primary.exists():
        roots.append(primary)
    if extra_datadirs:
        for d in extra_datadirs:
            p = Path(d)
            if p.exists() and p not in roots:
                roots.append(p)
    return roots


def assert_raw_ohlcv(
    userdir: Path | str,
    *,
    sample_limit: int = 5,
    skip_env: str = "AGENT_MARKET_NO_OHLCV_PREFLIGHT",
    extra_datadirs: Optional[Iterable[Path | str]] = None,
) -> None:
    """Raise ``OHLCVPreflightError`` if any OHLCV feather has != 6 cols or wrong order.

    Scans ``userdir/data/<exchange>/`` plus any ``extra_datadirs`` (e.g.
    `--datadir` values from a freqtrade invocation). Matches both spot
    pattern ``<PAIR>-<tf>.feather`` and futures pattern
    ``<PAIR>-<tf>-futures.feather``; deeper sub-namespaces (funding/,
    micro/, features/) are skipped to avoid false positives.

    No-op when ``${skip_env}=1`` is set in the environment, or when no
    candidate root exists (treated as "fresh checkout").
    """
    if os.environ.get(skip_env, "").strip() == "1":
        return
    udir = Path(userdir)
    roots = _candidate_data_roots(udir, extra_datadirs)
    if not roots:
        return

    def _is_ohlcv_feather(name: str) -> bool:
        return bool(_OHLCV_TF_RE.match(name) or _OHLCV_FUT_RE.match(name))

    contaminated: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        # Recursive glob: covers both `data/<exchange>/<pair>-<tf>.feather`
        # AND `data/<exchange>/futures/<pair>-<tf>-futures.feather`. Filename
        # regex still excludes funding / micro / features sub-namespaces.
        for feather_path in root.rglob("*.feather"):
            if feather_path in seen:
                continue
            seen.add(feather_path)
            if not _is_ohlcv_feather(feather_path.name):
                continue
            try:
                import pyarrow.feather as ft
                df = ft.read_feather(feather_path)
                cols = list(df.columns)
            except Exception:
                continue
            if len(cols) != 6 or tuple(cols[:6]) != _OHLCV:
                # Display path relative to userdir parent when possible
                try:
                    rel = str(feather_path.relative_to(udir.parent))
                except ValueError:
                    rel = str(feather_path)
                contaminated.append(rel)
                if len(contaminated) >= sample_limit:
                    break
        if len(contaminated) >= sample_limit:
            break

    if contaminated:
        raise OHLCVPreflightError(
            "freqtrade preflight failed: raw OHLCV feathers have non-standard "
            f"shape (expected 6 cols [date, open, high, low, close, volume]).\n"
            f"  contaminated examples: {contaminated[:3]}\n"
            "  cause: `factor_lab features mtf|xs|pair|funding|micro` writes "
            "engineered features back into the raw OHLCV path.\n"
            "  fix: run `python scripts/factor_lab.py features-restore "
            "{mtf|xs|pair|funding|micro|ohlcv_micro|microstructure}` to roll "
            "back from the .pre_<kind>.bak sidecars, or strip to 6 cols.\n"
            f"  override (per-shell): `export {skip_env}=1`."
        )


__all__ = ["OHLCVPreflightError", "assert_raw_ohlcv"]

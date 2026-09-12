from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from agent_market import paths


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in items:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _parse_timerange(raw: str | None) -> tuple[str, str] | None:
    value = str(raw or "").strip()
    if not value or "-" not in value:
        return None
    start, end = value.split("-", 1)
    if len(start) != 8 or len(end) != 8 or not start.isdigit() or not end.isdigit():
        return None
    return start, end


def _union_timerange(items: Iterable[str | None]) -> str:
    parsed = [part for item in items if (part := _parse_timerange(item))]
    if not parsed:
        return ""
    start = min(part[0] for part in parsed)
    end = max(part[1] for part in parsed)
    return f"{start}-{end}"


def _pair_slug(pair: str) -> str:
    return str(pair).replace("/", "_").replace(":", "_")


def _preferred_freqtrade_python() -> str:
    venv_python = paths.REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@dataclass(frozen=True)
class DataRequirements:
    miner_config_path: Path
    freqtrade_config_path: Path
    userdir: Path
    datadir: Path
    exchange: str
    trading_mode: str
    pairs: list[str]
    timeframes: list[str]
    timerange: str


def collect_data_requirements(config_path: str | Path) -> DataRequirements:
    miner_config_path = Path(config_path)
    if not miner_config_path.is_absolute():
        miner_config_path = paths.resolve_repo_path(miner_config_path)
    raw = _load_json(miner_config_path)

    backtest = raw.get("backtest") if isinstance(raw.get("backtest"), dict) else {}
    model_mining = raw.get("model_mining") if isinstance(raw.get("model_mining"), dict) else {}
    strategy_profile = raw.get("strategy_profile") if isinstance(raw.get("strategy_profile"), dict) else {}

    freqtrade_config_raw = (
        backtest.get("freqtrade_config")
        or raw.get("freqtrade_config")
        or ""
    )
    if not freqtrade_config_raw:
        raise ValueError(f"Missing freqtrade_config in {miner_config_path}")
    freqtrade_config_path = paths.resolve_repo_path(str(freqtrade_config_raw))
    ft_payload = _load_json(freqtrade_config_path)

    exchange = str(((ft_payload.get("exchange") or {}).get("name")) or "gate")
    trading_mode = str(ft_payload.get("trading_mode") or "spot")
    pairs = _unique(((ft_payload.get("exchange") or {}).get("pair_whitelist")) or [])
    base_timeframe = str(ft_payload.get("timeframe") or "1h")
    freqai_timeframes = list(
        (((ft_payload.get("freqai") or {}).get("feature_parameters") or {}).get("include_timeframes"))
        or []
    )
    allowed_informative = list(
        backtest.get("allowed_informative_timeframes")
        or strategy_profile.get("allowed_informative_timeframes")
        or raw.get("allowed_informative_timeframes")
        or []
    )
    max_strategy_timeframe = (
        backtest.get("max_strategy_timeframe")
        or strategy_profile.get("max_strategy_timeframe")
        or raw.get("max_strategy_timeframe")
        or ""
    )
    timeframes = _unique(
        [
            base_timeframe,
            *freqai_timeframes,
            *allowed_informative,
            str(max_strategy_timeframe or "").strip(),
        ]
    )

    timerange = _union_timerange(
        [
            raw.get("timerange"),
            raw.get("selection_timerange"),
            raw.get("holdout_timerange"),
            raw.get("train_timerange"),
            backtest.get("timerange"),
            backtest.get("selection_timerange"),
            backtest.get("holdout_timerange"),
            backtest.get("train_timerange"),
            model_mining.get("train_timerange"),
        ]
    )

    datadir = paths.resolve_repo_path(str(ft_payload.get("datadir") or "user_data/data"))
    userdir = datadir.parent if datadir.name == "data" else paths.user_data_root()

    return DataRequirements(
        miner_config_path=miner_config_path.resolve(),
        freqtrade_config_path=freqtrade_config_path.resolve(),
        userdir=userdir.resolve(),
        datadir=datadir.resolve(),
        exchange=exchange,
        trading_mode=trading_mode,
        pairs=pairs,
        timeframes=timeframes,
        timerange=timerange,
    )


def expected_ohlcv_paths(req: DataRequirements) -> list[Path]:
    out: list[Path] = []
    exchange_dir = req.datadir / req.exchange
    if req.trading_mode == "futures":
        futures_dir = exchange_dir / "futures"
        for pair in req.pairs:
            slug = _pair_slug(pair)
            for timeframe in req.timeframes:
                out.append(futures_dir / f"{slug}-{timeframe}-futures.feather")
    else:
        for pair in req.pairs:
            slug = _pair_slug(pair)
            for timeframe in req.timeframes:
                out.append(exchange_dir / f"{slug}-{timeframe}.feather")
    return out


def missing_ohlcv_paths(req: DataRequirements) -> list[Path]:
    return [path for path in expected_ohlcv_paths(req) if not path.exists()]


def build_proxy_env(proxy: str | None, base_env: Dict[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    if not proxy:
        return env
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[key] = proxy
    return env


def build_download_command(
    req: DataRequirements,
    *,
    python_executable: str | None = None,
) -> list[str]:
    cmd = [
        python_executable or _preferred_freqtrade_python(),
        str(paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"),
        "download-data",
        "--userdir",
        str(req.userdir),
        "--config",
        str(req.freqtrade_config_path),
        "--timeframes",
        *req.timeframes,
        "--pairs",
        *req.pairs,
        "--data-format-ohlcv",
        "feather",
    ]
    if req.timerange:
        cmd.extend(["--timerange", req.timerange])
    if req.trading_mode:
        cmd.extend(["--trading-mode", req.trading_mode])
    return cmd


def download_required_data(
    req: DataRequirements,
    *,
    proxy: str | None = None,
    check_only: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    missing_before = [str(path) for path in missing_ohlcv_paths(req)]
    cmd = build_download_command(req, python_executable=python_executable)
    if check_only or not missing_before:
        return {
            "command": cmd,
            "missing_before": missing_before,
            "missing_after": missing_before,
            "returncode": 0,
            "executed": False,
        }

    proc = subprocess.run(
        cmd,
        cwd=paths.REPO_ROOT,
        env=build_proxy_env(proxy),
        check=False,
        capture_output=True,
        text=True,
    )
    missing_after = [str(path) for path in missing_ohlcv_paths(req)]
    return {
        "command": cmd,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "returncode": int(proc.returncode),
        "executed": True,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


__all__ = [
    "DataRequirements",
    "build_download_command",
    "build_proxy_env",
    "collect_data_requirements",
    "download_required_data",
    "expected_ohlcv_paths",
    "missing_ohlcv_paths",
]

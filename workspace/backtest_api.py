"""Backtest API — wraps freqtrade backtesting as a single function call.

Usage:
    from workspace.backtest_api import run_backtest
    result = run_backtest(
        strategy_path="workspace/strategies/my_strategy.py",
        strategy_name="MyStrategy",
    )
    # result = {"ok": True, "sharpe": 0.72, ...} or {"ok": False, "error": "..."}
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
_FREQTRADE_CLI = ROOT / "scripts" / "freqtrade_cli.py"
_DEFAULT_CONFIG = ROOT / "workspace" / "configs" / "backtest_default.json"
_RESULTS_DIR = ROOT / "workspace" / "results"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _parse_backtest_zip(zip_path: Path) -> Dict[str, Any]:
    """Extract metrics from a freqtrade backtest result zip."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.endswith(".json") and "_config" not in n]
        if not members:
            return {"error": "no backtest JSON in zip"}
        data = json.loads(zf.read(members[0]))

    strategy_block = data.get("strategy") or {}
    if not strategy_block:
        return {"error": "empty strategy block"}

    strategy_name, metrics = next(iter(strategy_block.items()))
    trades = metrics.get("trades") or []

    # Extract strategy_comparison (has all the good metrics)
    comparison = data.get("strategy_comparison") or []
    comp = {}
    if comparison:
        comp = comparison[0] if isinstance(comparison[0], dict) else {}
    elif isinstance(metrics, dict):
        comp = metrics

    return {
        "ok": True,
        "strategy": str(strategy_name),
        "trades": int(comp.get("trades", len(trades))),
        "profit_pct": float(comp.get("profit_total_pct", 0)),
        "profit_abs": float(comp.get("profit_total_abs", 0)),
        "profit_factor": float(comp.get("profit_factor", 0)),
        "sharpe": float(comp.get("sharpe", 0)),
        "sortino": float(comp.get("sortino", 0)),
        "calmar": float(comp.get("calmar", 0)),
        "max_drawdown_pct": float(comp.get("max_drawdown_account", 0)) * 100,
        "max_drawdown_abs": float(comp.get("max_drawdown_abs", "0")),
        "win_rate": float(comp.get("winrate", 0)),
        "avg_duration": str(comp.get("duration_avg", "")),
        "cagr": float(comp.get("cagr", 0)),
        "sqn": float(comp.get("sqn", 0)),
        "expectancy": float(comp.get("expectancy", 0)),
    }


def run_backtest(
    strategy_path: str | Path,
    strategy_name: Optional[str] = None,
    *,
    config_path: Optional[str | Path] = None,
    pairs: Optional[List[str]] = None,
    timerange: str = "20251126-20260125",
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a freqtrade backtest and return standardized metrics.

    Args:
        strategy_path: Path to .py file containing the strategy class.
        strategy_name: Class name of the strategy. If None, auto-detected
                       from the file (first IStrategy subclass found).
        config_path: Path to freqtrade config JSON. Defaults to workspace default.
        pairs: Override pair whitelist. Defaults to config.
        timerange: Backtest time range (YYYYMMDD-YYYYMMDD).
        extra_args: Additional freqtrade CLI arguments.

    Returns:
        Dict with "ok": True and metrics, or "ok": False and "error" string.
    """
    strategy_path = Path(strategy_path).resolve()
    if not strategy_path.exists():
        return {"ok": False, "error": f"strategy file not found: {strategy_path}"}

    config_path = Path(config_path or _DEFAULT_CONFIG).resolve()
    if not config_path.exists():
        return {"ok": False, "error": f"config file not found: {config_path}"}

    # Auto-detect strategy name if not provided
    if strategy_name is None:
        strategy_name = _detect_strategy_name(strategy_path)
        if strategy_name is None:
            return {"ok": False, "error": f"no IStrategy subclass found in {strategy_path}"}

    # Build config with pair override if needed
    effective_config = config_path
    if pairs:
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            cfg["exchange"]["pair_whitelist"] = pairs
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=str(ROOT / "workspace" / "configs"))
            json.dump(cfg, tmp, indent=2)
            tmp.close()
            effective_config = Path(tmp.name)
        except Exception as exc:
            return {"ok": False, "error": f"config override failed: {exc}"}

    # Build command
    cmd = [
        sys.executable,
        str(_FREQTRADE_CLI),
        "backtesting",
        "--userdir", str(ROOT / "user_data"),
        "--config", str(effective_config),
        "--strategy", strategy_name,
        "--strategy-path", str(strategy_path.parent),
        "--timerange", timerange,
        "--cache", "none",
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Run backtest — clean PYTHONPATH to avoid freqtrade/ submodule shadowing.
    import os
    env = os.environ.copy()
    # Remove "." and project root from PYTHONPATH so site-packages freqtrade is used.
    # The strategy .py file injects its own sys.path at import time.
    old_pp = env.get("PYTHONPATH", "")
    clean_parts = [p for p in old_pp.split(os.pathsep) if p and p != "." and str(ROOT) not in p]
    env["PYTHONPATH"] = os.pathsep.join(clean_parts) if clean_parts else ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "backtest timed out (300s)"}
    except Exception as exc:
        return {"ok": False, "error": f"subprocess error: {exc}"}

    if proc.returncode != 0:
        # Extract useful error from stderr
        stderr = proc.stderr or ""
        # Look for Python traceback
        lines = stderr.strip().split("\n")
        error_lines = [l for l in lines[-10:] if l.strip()]
        error_msg = "\n".join(error_lines[-5:]) if error_lines else f"exit code {proc.returncode}"
        return {"ok": False, "error": error_msg, "stderr": stderr[-2000:]}

    # Find backtest result zip created AFTER we started the backtest
    # This avoids misattributing a stale zip from a concurrent/previous run.
    import time as _time
    backtest_start_time = _time.time() - 300  # allow up to 5 min
    results_dir = ROOT / "user_data" / "backtest_results"
    zips = [
        p for p in results_dir.glob("backtest-result-*.zip")
        if p.stat().st_mtime >= backtest_start_time
    ]
    zips.sort(key=lambda p: p.stat().st_mtime)
    if not zips:
        # Fallback: take newest regardless (backwards compat)
        zips = sorted(results_dir.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        return {"ok": False, "error": "no backtest result zip found"}

    latest_zip = zips[-1]
    try:
        result = _parse_backtest_zip(latest_zip)
    except Exception as exc:
        return {"ok": False, "error": f"failed to parse result: {exc}"}

    # Add metadata
    result["backtest_zip"] = str(latest_zip.relative_to(ROOT))
    result["strategy_path"] = str(strategy_path.relative_to(ROOT))
    result["strategy_sha256"] = _sha256(strategy_path)
    result["config_path"] = str(config_path.relative_to(ROOT))
    result["timerange"] = timerange
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Auto-save to results/
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = _RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Clean up temp config
    if effective_config != config_path and effective_config.exists():
        try:
            effective_config.unlink()
        except Exception:
            pass

    return result


def _detect_strategy_name(path: Path) -> Optional[str]:
    """Find the first IStrategy subclass name in a .py file."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "IStrategy":
                    return node.name
    return None


__all__ = ["run_backtest"]

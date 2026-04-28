"""Unified walk-forward backtest (training + freqtrade backtesting)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .paths import ROOT, DEFAULT_PAIRS, DEFAULT_LABEL_PERIOD, DEFAULT_CLASS_THRESHOLD, MODELS_DIR


DEFAULT_BASE_CFG = {
    "data": {
        "feature_file": "user_data/freqai_features_real.json",
        "expressions_file": "user_data/freqai_expressions.json",
        "data_dir": "user_data/data", "exchange": "kucoin", "timeframe": "1h",
        "label_period": DEFAULT_LABEL_PERIOD, "task": "classify_3way",
        "class_threshold": DEFAULT_CLASS_THRESHOLD,
        "pairs": DEFAULT_PAIRS,
    },
    "model": {"name": "lightgbm", "params": {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "n_estimators": 200, "num_boost_round": 200, "learning_rate": 0.02,
        "num_leaves": 15, "min_child_samples": 150, "subsample": 0.6,
        "colsample_bytree": 0.4, "reg_alpha": 3.0, "reg_lambda": 3.0,
        "max_depth": 4, "verbosity": -1,
    }},
    "training": {"validation_ratio": 0.2, "purge": 12, "embargo": 6},
    "output": {},
}


def _fmt(d: datetime) -> str: return d.strftime("%Y%m%d")


def generate_windows(train_months: int = 6, test_days: int = 30, step_days: int = 30,
                     data_start: datetime = datetime(2025, 7, 1),
                     data_end: datetime = datetime(2026, 4, 12),
                     test_start_anchor: Optional[datetime] = None,
                     num_windows: Optional[int] = None) -> List[Dict]:
    """Default data_start = TRAIN3 end (2025-07-01). Walk-forward's train
    window begins here, so no factor-selection-period data leaks into training.
    Set data_start earlier only when explicitly running legacy/diagnostic tests.
    """
    out = []
    ts = data_start
    if test_start_anchor is None:
        # With data_start = 2025-07-01 and 6-month train, first test starts 2026-01.
        test_start_anchor = datetime(2026, 1, 1)
    if train_months >= 12:
        for i in range(7):
            test_start = test_start_anchor + timedelta(days=i * step_days)
            test_end = test_start + timedelta(days=test_days)
            if test_end > data_end: test_end = data_end
            if test_start >= data_end: break
            train_end = test_start
            train_start = train_end - timedelta(days=train_months * 30)
            if train_start < datetime(2023, 5, 15):
                train_start = datetime(2023, 5, 15)
            out.append({"train_start": train_start, "train_end": train_end,
                        "test_start": test_start, "test_end": test_end})
    else:
        # Optional anchor override: pin the FIRST test_start and back-compute
        # train_start so train_end==test_start. This is the path used by
        # "single-window" or "late-period" diagnostic runs.
        if test_start_anchor is not None and test_start_anchor > data_start:
            vs = test_start_anchor
            while True:
                ts_w = vs - timedelta(days=train_months * 30)
                te = vs
                ve = vs + timedelta(days=test_days)
                if ve > data_end: ve = data_end
                if vs >= data_end: break
                out.append({"train_start": ts_w, "train_end": te,
                            "test_start": vs, "test_end": ve})
                if num_windows is not None and len(out) >= num_windows:
                    break
                vs = vs + timedelta(days=step_days)
        else:
            while True:
                te = ts + timedelta(days=train_months * 30)
                vs = te
                ve = vs + timedelta(days=test_days)
                if ve > data_end: ve = data_end
                if vs >= data_end: break
                out.append({"train_start": ts, "train_end": te,
                            "test_start": vs, "test_end": ve})
                if num_windows is not None and len(out) >= num_windows:
                    break
                ts = ts + timedelta(days=step_days)
    return out


def train_window(w: Dict, tag: str, base_cfg: Dict = None,
                  exit_label_period: Optional[int] = None) -> Optional[Path]:
    """Train entry model. If exit_label_period given, also train a second (exit)
    model with a shorter horizon, saved to <model_dir>_exit. Strategy reads both
    via AGENT_MODEL_DIR + AGENT_EXIT_MODEL_DIR."""
    base_cfg = base_cfg or DEFAULT_BASE_CFG
    model_dir = MODELS_DIR / tag
    done = (model_dir / "training_summary.json").exists()
    exit_dir = MODELS_DIR / f"{tag}_exit"
    exit_done = (exit_dir / "training_summary.json").exists() if exit_label_period else True
    if done and exit_done:
        return model_dir
    # ----- Entry model -----
    if not done:
        model_dir.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(base_cfg))
        cfg["data"]["train_timerange"] = f"{_fmt(w['train_start'])}-{_fmt(w['train_end'])}"
        cfg["output"]["model_dir"] = str(model_dir)
        cfg_path = ROOT / "configs" / f"_lab_{tag}.json"
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        p = subprocess.run(
            [sys.executable, "scripts/train_pipeline.py", "--config", str(cfg_path)],
            capture_output=True, text=True, cwd=ROOT, timeout=1500,
        )
        cfg_path.unlink(missing_ok=True)
        if not (model_dir / "training_summary.json").exists():
            print(f"  [FAIL entry] {p.stderr[-300:]}"); return None
    # ----- Exit model (shorter horizon) -----
    if exit_label_period and not exit_done:
        exit_dir.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(base_cfg))
        cfg["data"]["train_timerange"] = f"{_fmt(w['train_start'])}-{_fmt(w['train_end'])}"
        cfg["data"]["label_period"] = int(exit_label_period)
        cfg["output"]["model_dir"] = str(exit_dir)
        cfg_path = ROOT / "configs" / f"_lab_{tag}_exit.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        p = subprocess.run(
            [sys.executable, "scripts/train_pipeline.py", "--config", str(cfg_path)],
            capture_output=True, text=True, cwd=ROOT, timeout=1500,
        )
        cfg_path.unlink(missing_ok=True)
        if not (exit_dir / "training_summary.json").exists():
            print(f"  [WARN exit model FAIL] {p.stderr[-300:]}")
    return model_dir


def backtest_window(w: Dict, model_dir: Path, strategy: str = "ELExitATRLSCls",
                     ft_config: str = "user_data/config_okx_futures_backtest.json",
                     exit_model_dir: Optional[Path] = None,
                     datadir: Optional[str] = None) -> Optional[Dict]:
    tr = f"{_fmt(w['test_start'])}-{_fmt(w['test_end'])}"
    env = os.environ.copy()
    env["AGENT_MODEL_DIR"] = str(model_dir)
    if exit_model_dir and (exit_model_dir / "training_summary.json").exists():
        env["AGENT_EXIT_MODEL_DIR"] = str(exit_model_dir)
    cmd = ["freqtrade", "backtesting", "--config", ft_config,
           "--strategy", strategy, "--timerange", tr]
    if datadir:
        cmd += ["--datadir", datadir]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)
    out = p.stdout + p.stderr
    t = re.search(r"Total/Daily Avg Trades[^\d]*(\d+)\s*/", out)
    profit = None
    for line in out.split("\n"):
        if "Total profit %" in line:
            nums = re.findall(r"-?\d+\.\d+", line)
            if nums: profit = float(nums[0]); break
    w_ = re.search(r"Win\s+Draw\s+Loss\s+Win%.*?(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)", out, re.DOTALL)
    dd = re.search(r"Max % of account underwater.*?(-?[\d.]+)%", out)
    mkt = re.search(r"Market change.*?(-?[\d.]+)%", out)
    if t is None or profit is None: return None
    return {"trades": int(t.group(1)), "profit_pct": profit,
            "win_pct": float(w_.group(4)) if w_ else 0.0,
            "dd_pct": float(dd.group(1)) if dd else 0.0,
            "market_chg": float(mkt.group(1)) if mkt else 0.0}


def run_walkforward(tag: str, train_months: int = 6, strategy: str = "ELExitATRLSCls",
                    ft_config: str = "user_data/config_okx_futures_backtest.json",
                    model_name: str = "lightgbm",
                    model_params: Optional[Dict] = None,
                    test_start_anchor: Optional[datetime] = None,
                    num_windows: Optional[int] = None,
                    expressions_file: Optional[str] = None,
                    exit_label_period: Optional[int] = None,
                    datadir: Optional[str] = None,
                    **kwargs) -> List[Dict]:
    windows = generate_windows(train_months=train_months,
                                test_start_anchor=test_start_anchor,
                                num_windows=num_windows, **kwargs)
    print(f"[walkforward] tag={tag}, windows={len(windows)}, train_months={train_months}, model={model_name}")
    if expressions_file:
        print(f"[walkforward] expressions_file={expressions_file}")
    # Allow caller to override the default LightGBM model / expressions for walk-forward
    base_cfg = None
    if model_name != "lightgbm" or model_params is not None or expressions_file:
        base_cfg = json.loads(json.dumps(DEFAULT_BASE_CFG))
        base_cfg["model"]["name"] = model_name
        if model_params:
            base_cfg["model"]["params"] = model_params
        if expressions_file:
            base_cfg["data"]["expressions_file"] = expressions_file
    results = []
    for i, w in enumerate(windows):
        window_tag = f"{tag}_{_fmt(w['train_end'])}"
        print(f"\n[{i+1}/{len(windows)}] {window_tag}")
        md = train_window(w, window_tag, base_cfg=base_cfg, exit_label_period=exit_label_period)
        if md is None: continue
        s = json.loads((md / "training_summary.json").read_text())
        m = s.get("metrics", {})
        print(f"  train_size={s.get('train_size','?')} acc_v={m.get('accuracy_valid', 0):.3f}")
        exit_md = MODELS_DIR / f"{window_tag}_exit" if exit_label_period else None
        bt = backtest_window(w, md, strategy=strategy, ft_config=ft_config,
                             exit_model_dir=exit_md, datadir=datadir)
        if bt is None: print("  BACKTEST FAIL"); continue
        print(f"  profit={bt['profit_pct']:+.2f}% win={bt['win_pct']:.1f}% "
              f"dd={bt['dd_pct']:.2f}% trades={bt['trades']} mkt={bt['market_chg']:+.1f}%")
        results.append({"window": i+1, "test": f"{_fmt(w['test_start'])}-{_fmt(w['test_end'])}",
                        "acc_v": m.get("accuracy_valid", 0),
                        "train_size": s.get("train_size", 0), **bt})

    if results:
        tp = sum(r["profit_pct"] for r in results)
        tn = sum(r["trades"] for r in results)
        aw = sum(r["win_pct"] * r["trades"] for r in results) / max(tn, 1)
        md_ = max(r["dd_pct"] for r in results)
        print(f"\n{tag}: profit={tp:+.2f}% trades={tn} avg_win={aw:.1f}% max_dd={md_:.2f}%")
    return results

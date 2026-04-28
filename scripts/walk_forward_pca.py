#!/usr/bin/env python3
"""Walk-forward: honest_v12 vs honest_v12 + PCA(95%) on same 4 windows."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

STRATEGY = "ELExitATRLSClsLong"
FT_CONFIG = "user_data/config_freqai_kucoin.json"
TRAIN_MONTHS = 6
TEST_DAYS = 30
STEP_DAYS = 30
TEST_START = datetime(2026, 1, 1)
TEST_END = datetime(2026, 4, 12)

BASE_CFG = {
    "data": {
        "feature_file": "user_data/freqai_features_real.json",
        "expressions_file": "user_data/freqai_expressions_honest_v12.json",
        "data_dir": "user_data/data",
        "exchange": "kucoin",
        "timeframe": "1h",
        "label_period": 12,
        "task": "classify_3way",
        "class_threshold": 0.005,
        "pairs": [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        ],
    },
    "model": {
        "name": "lightgbm",
        "params": {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "num_boost_round": 200,
            "learning_rate": 0.02,
            "num_leaves": 15,
            "min_child_samples": 150,
            "subsample": 0.6,
            "colsample_bytree": 0.4,
            "reg_alpha": 3.0,
            "reg_lambda": 3.0,
            "max_depth": 4,
            "verbosity": -1,
        },
    },
    "training": {"validation_ratio": 0.2, "purge": 12, "embargo": 6},
    "output": {},
}


def _fmt(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def generate_windows() -> list[dict]:
    out = []
    ts = TEST_START - timedelta(days=TRAIN_MONTHS * 30)  # 6 months before first test
    while True:
        te = ts + timedelta(days=TRAIN_MONTHS * 30)
        vs = te
        ve = min(vs + timedelta(days=TEST_DAYS), TEST_END)
        if vs >= TEST_END:
            break
        out.append({
            "train_start": ts, "train_end": te,
            "test_start": vs, "test_end": ve,
        })
        ts = ts + timedelta(days=STEP_DAYS)
    return out


def _train(tag: str, w: dict, extra_training: dict | None = None) -> Path | None:
    model_dir = ROOT / "artifacts" / "models" / tag
    if (model_dir / "training_summary.json").exists():
        print(f"    [SKIP] {tag} already trained")
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["data"]["train_timerange"] = f"{_fmt(w['train_start'])}-{_fmt(w['train_end'])}"
    cfg["output"]["model_dir"] = str(model_dir)
    if extra_training:
        cfg["training"].update(extra_training)
    cfg_path = ROOT / "configs" / f"_wfpca_{tag}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/train_pipeline.py", "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=ROOT, timeout=900,
    )
    cfg_path.unlink(missing_ok=True)
    if not (model_dir / "training_summary.json").exists():
        print(f"    [FAIL] {proc.stderr[-400:]}")
        return None
    return model_dir


def _backtest(w: dict, model_dir: Path) -> dict | None:
    tr = f"{_fmt(w['test_start'])}-{_fmt(w['test_end'])}"
    env = os.environ.copy()
    env["AGENT_MODEL_DIR"] = str(model_dir)
    data_dir = str(ROOT / "user_data" / "data" / "data_clean" / "kucoin")
    proc = subprocess.run(
        ["freqtrade", "backtesting", "--config", FT_CONFIG,
         "--strategy", STRATEGY, "--timerange", tr, "--cache", "none",
         "--datadir", data_dir],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=600,
    )
    out = proc.stdout + proc.stderr
    t = re.search(r"Total/Daily Avg Trades[^\d]*(\d+)\s*/", out)
    profit = None
    for line in out.split("\n"):
        if "Total profit %" in line:
            nums = re.findall(r"-?\d+\.\d+", line)
            if nums:
                profit = float(nums[0])
                break
    w_ = re.search(r"Win\s+Draw\s+Loss\s+Win%.*?(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)", out, re.DOTALL)
    dd = re.search(r"Max % of account underwater.*?(-?[\d.]+)%", out)
    mkt = re.search(r"Market change.*?(-?[\d.]+)%", out)
    if t is None or profit is None:
        # Check for 0-trades result (freqtrade outputs the table but profit line may be absent)
        zero_trades = re.search(r"ELExitATRLSClsLong.{1,20}0.{1,20}0\.00", out)
        if zero_trades:
            mkt_val = float(mkt.group(1)) if mkt else 0.0
            return {"trades": 0, "profit_pct": 0.0, "win_pct": 0.0, "dd_pct": 0.0, "market_chg": mkt_val}
        print(f"    [BACKTEST FAIL] stdout tail:\n{out[-500:]}")
        return None
    return {
        "trades": int(t.group(1)),
        "profit_pct": profit,
        "win_pct": float(w_.group(4)) if w_ else 0.0,
        "dd_pct": float(dd.group(1)) if dd else 0.0,
        "market_chg": float(mkt.group(1)) if mkt else 0.0,
    }


def run_variant(label: str, tag_prefix: str, windows: list[dict],
                extra_training: dict | None = None) -> list[dict]:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    results = []
    for i, w in enumerate(windows):
        tag = f"{tag_prefix}_{_fmt(w['train_end'])}"
        print(f"\n[{i+1}/{len(windows)}] test {_fmt(w['test_start'])}→{_fmt(w['test_end'])} | tag={tag}")
        md = _train(tag, w, extra_training)
        if md is None:
            continue
        s = json.loads((md / "training_summary.json").read_text())
        m = s.get("metrics", {})
        pca_info = s.get("pca", {})
        acc_v = m.get("accuracy_valid", 0)
        if pca_info:
            print(f"  acc_v={acc_v:.3f} | PCA: {pca_info['n_original']} → {pca_info['n_components']} components ({pca_info['explained_variance_pct']:.1f}%)")
        else:
            print(f"  acc_v={acc_v:.3f} | features={len(s.get('features', []))}")
        bt = _backtest(w, md)
        if bt is None:
            continue
        print(f"  profit={bt['profit_pct']:+.2f}% win={bt['win_pct']:.1f}% "
              f"dd={bt['dd_pct']:.2f}% trades={bt['trades']} mkt={bt['market_chg']:+.1f}%")
        results.append({
            "window": i + 1,
            "test": f"{_fmt(w['test_start'])}-{_fmt(w['test_end'])}",
            "acc_v": acc_v,
            **bt,
        })
    return results


def summarize(label: str, results: list[dict]) -> None:
    if not results:
        print(f"\n{label}: no results")
        return
    total = sum(r["profit_pct"] for r in results)
    tr_n = sum(r["trades"] for r in results)
    avg_win = sum(r["win_pct"] * r["trades"] for r in results) / max(tr_n, 1)
    max_dd = max(r["dd_pct"] for r in results)
    print(f"\n{label}: profit={total:+.2f}% trades={tr_n} avg_win={avg_win:.1f}% max_dd={max_dd:.2f}%")


def main() -> int:
    windows = generate_windows()
    print(f"Walk-forward PCA comparison: {len(windows)} windows")
    for i, w in enumerate(windows):
        print(f"  W{i+1}: train {_fmt(w['train_start'])}→{_fmt(w['train_end'])} | "
              f"test {_fmt(w['test_start'])}→{_fmt(w['test_end'])}")

    # Variant A: v12 baseline (reuse existing honest_v12 models if present)
    res_v12 = run_variant(
        "honest_v12 (baseline, no PCA)",
        "honest_v12",
        windows,
        extra_training=None,
    )

    # Variant B: v12 + PCA 95%
    res_pca = run_variant(
        "honest_v12_pca95 (PCA n_components=0.95)",
        "honest_v12_pca95",
        windows,
        extra_training={"pca": {"n_components": 0.95}},
    )

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'W':<3} {'Test':<22} {'v12':>8} {'pca95':>8} {'delta':>8} {'v12_win':>8} {'pca_win':>8}")
    print("=" * 80)
    for i in range(max(len(res_v12), len(res_pca))):
        r1 = res_v12[i] if i < len(res_v12) else None
        r2 = res_pca[i] if i < len(res_pca) else None
        test = (r1 or r2)["test"]
        p1 = f"{r1['profit_pct']:+.2f}%" if r1 else "  FAIL"
        p2 = f"{r2['profit_pct']:+.2f}%" if r2 else "  FAIL"
        delta = f"{(r2['profit_pct'] - r1['profit_pct']):+.2f}%" if r1 and r2 else "    —"
        w1 = f"{r1['win_pct']:.1f}%" if r1 else "    —"
        w2 = f"{r2['win_pct']:.1f}%" if r2 else "    —"
        print(f"{i+1:<3} {test:<22} {p1:>8} {p2:>8} {delta:>8} {w1:>8} {w2:>8}")
    print("=" * 80)

    summarize("honest_v12 (no PCA)", res_v12)
    summarize("honest_v12_pca95", res_pca)

    out = {"v12": res_v12, "v12_pca95": res_pca}
    out_path = ROOT / "artifacts" / "walk_forward_pca_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

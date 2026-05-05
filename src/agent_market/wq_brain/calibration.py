"""local↔remote calibration for the local-simulate gate.

Background
----------

The mandatory local-simulate gate (added in Phase 3 of agent_brief) drops
candidates whose ``wq_fitness < 0.5``. That 0.5 threshold was inherited
from QuantGPT and tuned for a different universe; on Russell 2070 (Kaggle)
predicting WQ TOP3000 ``fitness ≥ 1.0`` it could be too tight or too loose.

This module's job is to do an end-to-end **calibration**:

  1. Read the historical ``tried_exprs.jsonl`` ledger (one row per remote
     simulate). Pick the top-N best-fit alphas with non-null sharpe/fitness.
  2. Re-run each one through ``simulate_expression_locally`` to get a
     (local_fitness, local_sharpe, local_turnover) tuple side-by-side with
     the remote result.
  3. Report Pearson + Spearman correlation between local & remote fitness,
     and search for the ``local_fitness`` cut-point that maximises F1 on
     "remote fitness ≥ 1.0" (i.e. would-pass-the-gate predictor).
  4. Persist the recommended threshold under
     ``artifacts/wq_brain/calibration/{tag}/threshold.json`` so
     ``local_sim._wq_rating`` can pick it up automatically.

Cost: local-simulate on 13M-row OHLCV is ~1-2 minutes per expression, so
30 samples ≈ 30-60 minutes. Designed to be run as an offline batch.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .local_sim import simulate_expression_locally
from .paths import tried_exprs_path, wq_brain_root
from .tried_log import read_tried

logger = logging.getLogger(__name__)


# ── Filesystem layout ───────────────────────────────────────────────────


def calibration_root(tag: str) -> Path:
    return wq_brain_root() / "calibration" / tag


def threshold_path(tag: str) -> Path:
    return calibration_root(tag) / "threshold.json"


def samples_path(tag: str) -> Path:
    return calibration_root(tag) / "samples.jsonl"


def report_path(tag: str) -> Path:
    return calibration_root(tag) / "report.md"


# ── Threshold accessor (used by local_sim) ──────────────────────────────


def load_calibrated_threshold(tag: str, *, default: float = 0.5) -> float:
    """Return the calibrated local-fitness gate threshold for ``tag``.

    Falls back to ``default`` (the legacy 0.5) when no calibration has been
    run yet.
    """
    p = threshold_path(tag)
    if not p.exists():
        return default
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        v = data.get("threshold")
        if v is None:
            return default
        return float(v)
    except (ValueError, OSError, KeyError):
        return default


def save_calibrated_threshold(tag: str, threshold: float, *, metadata: Optional[dict] = None) -> Path:
    """Write the threshold + metadata atomically."""
    p = threshold_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": float(threshold), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if metadata:
        payload["metadata"] = metadata
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


# ── Statistics ──────────────────────────────────────────────────────────


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def _rank(values: list[float]) -> list[float]:
    """Average-rank vector (1-based), handles ties with mid-rank."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        # Walk through ties on value
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based midrank
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson(_rank(xs), _rank(ys))


def _rmse(xs: list[float], ys: list[float]) -> Optional[float]:
    if not xs or len(xs) != len(ys):
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(xs, ys)) / len(xs))


def find_best_threshold(
    samples: list[dict[str, Any]],
    *,
    target_label: str = "passes_remote",
    score_field: str = "local_fitness",
) -> dict[str, Any]:
    """Sweep candidate thresholds on local_fitness, pick the F1-maximising one.

    Each sample needs ``local_fitness`` and a boolean ``passes_remote``
    (remote fitness ≥ 1.0 by convention). Returns dict with chosen
    threshold + precision / recall / F1 at it.
    """
    valid = [s for s in samples if s.get(score_field) is not None]
    if len(valid) < 5:
        return {
            "threshold": 0.5,
            "method": "fallback (insufficient samples)",
            "n_samples": len(valid),
            "precision": None,
            "recall": None,
            "f1": None,
        }
    locals_ = sorted({float(s[score_field]) for s in valid})
    # candidates = midpoints between sorted unique local_fitness values
    candidates = sorted({0.0, 0.3, 0.5, 0.75, 1.0} |
                        {(a + b) / 2 for a, b in zip(locals_, locals_[1:])})
    pos_total = sum(1 for s in valid if s.get(target_label))
    if pos_total == 0:
        return {
            "threshold": 0.5,
            "method": "fallback (no positives in sample)",
            "n_samples": len(valid),
            "precision": None,
            "recall": None,
            "f1": None,
        }

    best: dict[str, Any] = {"threshold": 0.5, "f1": -1.0,
                             "precision": 0.0, "recall": 0.0,
                             "tp": 0, "fp": 0, "fn": 0,
                             "method": "max-f1 sweep"}
    for thr in candidates:
        tp = sum(1 for s in valid if s.get(target_label) and s[score_field] >= thr)
        fp = sum(1 for s in valid if not s.get(target_label) and s[score_field] >= thr)
        fn = sum(1 for s in valid if s.get(target_label) and s[score_field] < thr)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tp": tp, "fp": fp, "fn": fn,
                "n_samples": len(valid),
                "method": "max-f1 sweep",
            }
    return best


# ── Sample selection from tried_log ─────────────────────────────────────


def _sample_score(row: dict[str, Any]) -> float:
    """Use remote fitness as the priority key when picking calibration samples."""
    fi = row.get("fitness")
    return float(fi) if isinstance(fi, (int, float)) else float("-inf")


def select_calibration_samples(
    tag: str,
    *,
    top_n: int = 30,
    require_alpha_id: bool = False,
) -> list[dict[str, Any]]:
    """Pick the top-N COMPLETE rows from tried_log with non-null metrics.

    Sorted by remote fitness desc so we prefer samples near the gate.
    """
    rows = read_tried(tried_exprs_path(tag), tail=2000)
    valid: list[dict[str, Any]] = []
    seen_exprs: set[str] = set()
    for r in rows:
        if (r.get("status") != "COMPLETE"
                or r.get("sharpe") is None
                or r.get("fitness") is None
                or r.get("turnover") is None):
            continue
        if require_alpha_id and not r.get("alpha_id"):
            continue
        expr = (r.get("expr") or "").strip()
        if not expr or expr in seen_exprs:
            continue
        seen_exprs.add(expr)
        valid.append(r)
    valid.sort(key=_sample_score, reverse=True)
    return valid[:top_n]


# ── Main calibration entry ──────────────────────────────────────────────


@dataclass
class CalibrationResult:
    tag: str
    samples: list[dict[str, Any]]
    pearson_fitness: Optional[float]
    spearman_fitness: Optional[float]
    rmse_fitness: Optional[float]
    threshold: dict[str, Any]
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "n_samples": len(self.samples),
            "pearson_fitness": self.pearson_fitness,
            "spearman_fitness": self.spearman_fitness,
            "rmse_fitness": self.rmse_fitness,
            "threshold": self.threshold,
            "elapsed_sec": round(self.elapsed_sec, 1),
        }


def calibrate_local_threshold(
    tag: str,
    *,
    top_n: int = 30,
    max_time_sec: Optional[float] = None,
    save: bool = True,
    pass_remote_fitness: float = 1.0,
    progress_cb: Optional[Any] = None,
) -> CalibrationResult:
    """End-to-end: select samples → re-simulate locally → pick threshold.

    ``progress_cb(i, total, sample)`` is called between simulations so the
    CLI can stream progress (~1-2 min/sample on Russell 2070).
    """
    rows = select_calibration_samples(tag, top_n=top_n)
    if len(rows) < 5:
        raise RuntimeError(
            f"calibration needs at least 5 COMPLETE samples in tried_exprs; "
            f"only {len(rows)} found for tag={tag!r}"
        )
    samples_p = samples_path(tag)
    samples_p.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    samples: list[dict[str, Any]] = []
    with samples_p.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if max_time_sec is not None and (time.time() - t0) > max_time_sec:
                logger.warning("calibration time budget exhausted at %d/%d", i - 1, len(rows))
                break
            expr = row["expr"]
            if progress_cb is not None:
                progress_cb(i, len(rows), expr)
            try:
                local = simulate_expression_locally(expr)
            except Exception as exc:
                logger.warning("local-simulate failed for %r: %s", expr[:60], exc)
                continue
            sample = {
                "expr": expr,
                "alpha_id": row.get("alpha_id"),
                "remote_sharpe": float(row["sharpe"]),
                "remote_fitness": float(row["fitness"]),
                "remote_turnover": float(row["turnover"]),
                "local_sharpe": float(local.wq_sharpe),
                "local_fitness": float(local.wq_fitness),
                "local_turnover": float(local.wq_turnover),
                "passes_remote": float(row["fitness"]) >= pass_remote_fitness,
            }
            samples.append(sample)
            f.write(json.dumps(sample) + "\n")
            f.flush()
    elapsed = time.time() - t0

    remote_fi = [s["remote_fitness"] for s in samples]
    local_fi = [s["local_fitness"] for s in samples]
    pearson = _pearson(remote_fi, local_fi)
    spearman = _spearman(remote_fi, local_fi)
    rmse = _rmse(remote_fi, local_fi)
    thr = find_best_threshold(samples)

    if save and len(samples) >= 5:
        save_calibrated_threshold(
            tag, thr["threshold"],
            metadata={
                "n_samples": len(samples),
                "pearson_fitness": pearson,
                "spearman_fitness": spearman,
                "rmse_fitness": rmse,
                "f1": thr.get("f1"),
                "precision": thr.get("precision"),
                "recall": thr.get("recall"),
                "method": thr.get("method"),
            },
        )
        _write_report(tag, samples, pearson, spearman, rmse, thr, elapsed)

    return CalibrationResult(
        tag=tag, samples=samples,
        pearson_fitness=pearson, spearman_fitness=spearman, rmse_fitness=rmse,
        threshold=thr, elapsed_sec=elapsed,
    )


def _write_report(
    tag: str,
    samples: list[dict[str, Any]],
    pearson: Optional[float],
    spearman: Optional[float],
    rmse: Optional[float],
    thr: dict[str, Any],
    elapsed: float,
) -> None:
    lines = [
        f"# Local↔Remote Calibration — `{tag}`",
        "",
        f"_n_samples = {len(samples)} · elapsed = {elapsed:.1f}s_",
        "",
        "## Correlation",
        "",
        f"- Pearson(remote_fitness, local_fitness): {pearson:.3f}" if pearson is not None else "- Pearson: n/a",
        f"- Spearman(remote_fitness, local_fitness): {spearman:.3f}" if spearman is not None else "- Spearman: n/a",
        f"- RMSE: {rmse:.3f}" if rmse is not None else "- RMSE: n/a",
        "",
        "## Recommended threshold",
        "",
        f"- threshold (local_fitness ≥ X → predicted to pass remote fi ≥ 1.0):"
        f" **{thr['threshold']:.3f}**",
        f"- precision: {thr.get('precision')}",
        f"- recall: {thr.get('recall')}",
        f"- f1: {thr.get('f1')}",
        f"- method: {thr.get('method')}",
        "",
        "## Samples",
        "",
        "| # | remote_fi | local_fi | passes | expr |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(samples[:30], 1):
        expr = (s.get("expr") or "")[:80].replace("|", "/")
        lines.append(
            f"| {i} | {s['remote_fitness']:.2f} | {s['local_fitness']:.2f} | "
            f"{'✓' if s['passes_remote'] else '✗'} | `{expr}` |"
        )
    report_path(tag).write_text("\n".join(lines), encoding="utf-8")

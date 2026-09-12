"""Non-LLM batch scan: expand seed templates × grid → simulate → filter → submit."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .client import WQSession, session_from_env
from .dtypes import AlphaCandidate, AlphaSettings
from .operators import validate_expression
from .paths import alpha_pool_path, tried_exprs_path, wq_brain_run_dir
from .pool import AlphaPool, DuplicateDetector
from .seeds import expand_all, load_seeds_config
from .tried_log import append_tried

logger = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    tag: str
    seed_file: Path
    region: str = "USA"
    universe: str = "TOP3000"
    decay: int = 6
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    max_candidates: int = 200
    auto_submit: bool = False
    legacy_unsafe_auto_submit: bool = False  # Codex R2: gate the bypass
    quality_sharpe_min: float = 1.25
    quality_fitness_min: float = 1.0
    dry_run: bool = False


def _make_run_id(tag: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"wqbrain_scan_{tag}_{ts}_{uuid.uuid4().hex[:8]}"


def run_scan(config: ScanConfig, session: Optional[WQSession] = None) -> dict:
    seeds, grid = load_seeds_config(Path(config.seed_file))
    raw_exprs = expand_all(seeds, grid)
    logger.info("Scan: %d templates × grid → %d raw exprs", len(seeds), len(raw_exprs))

    valid: list[str] = []
    invalid_count = 0
    for e in raw_exprs:
        if validate_expression(e):
            invalid_count += 1
        else:
            valid.append(e)
    logger.info("Scan: %d valid / %d invalid", len(valid), invalid_count)

    detector = DuplicateDetector()
    unique: list[str] = []
    for e in valid:
        if not detector.is_duplicate(e):
            detector.add(e)
            unique.append(e)
    logger.info("Scan: %d unique after dedup", len(unique))

    unique = unique[: config.max_candidates]

    settings = AlphaSettings(
        region=config.region,
        universe=config.universe,
        decay=config.decay,
        neutralization=config.neutralization,
        truncation=config.truncation,
    )
    candidates = [AlphaCandidate(expr=e, settings=settings, source="scan") for e in unique]

    run_id = _make_run_id(config.tag)
    run_dir = wq_brain_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "seed_file": str(config.seed_file)}, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "expansion.json").write_text(
        json.dumps(
            {
                "raw_count": len(raw_exprs),
                "valid_count": len(valid),
                "unique_count": len(unique),
                "exprs": unique,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if config.dry_run:
        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "candidates": len(unique),
            "simulated": 0,
            "passed": 0,
            "submitted": 0,
            "dry_run": True,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    sess = session or session_from_env()
    logger.info("Scan: simulating %d candidates", len(candidates))
    candidates = sess.batch_simulate(candidates)

    (run_dir / "sim_results.json").write_text(
        json.dumps([c.to_dict() for c in candidates], indent=2), encoding="utf-8"
    )

    passed = [
        c for c in candidates
        if c.sim_result
        and c.sim_result.passes_quality(
            sharpe_min=config.quality_sharpe_min,
            fitness_min=config.quality_fitness_min,
        )
    ]
    logger.info("Scan: %d/%d passed quality gate", len(passed), len(candidates))

    # Append every simulated candidate to cross-loop tried log
    tried_path = tried_exprs_path(config.tag)
    for c in candidates:
        r = c.sim_result
        if r is None:
            continue
        try:
            # Seed-driven scan candidates: no parent (root of search tree),
            # altitude L1 (region/universe-level exploration), evidence_type
            # 'seed'. ``delta_U`` is omitted because there's no parent to diff
            # against; cache scoring falls back to recency.
            append_tried(
                tried_path, expr=c.expr,
                sharpe=r.sharpe, fitness=r.fitness, turnover=r.turnover,
                alpha_id=r.alpha_id, status=r.status, error=r.error,
                region=config.region, universe=config.universe, decay=config.decay,
                evidence_type="seed",
                altitude="L1_region_universe",
            )
        except Exception as exc:
            logger.warning("tried_log append failed for %s: %s", c.candidate_id, exc)

    pool = AlphaPool(alpha_pool_path(config.tag))
    submitted_count = 0
    # Codex review R2 (remote readiness): `auto_submit` here bypasses
    # `pool submit-worker`'s quota-reserve / local-jaccard / self-corr /
    # outcome-persistence stack. Refuse unless caller explicitly opted in
    # with `legacy_unsafe_auto_submit=True`. The CLI exposes this via
    # `--auto-submit --legacy-unsafe`. Production users should run
    # `pool submit-worker` after `scan` instead.
    auto_submit_refused = (
        config.auto_submit
        and not getattr(config, "legacy_unsafe_auto_submit", False)
    )
    if auto_submit_refused:
        logger.warning(
            "scan auto-submit refused: legacy unsafe path. Re-run with "
            "`--auto-submit --legacy-unsafe` to opt in, or "
            "preferred: drop --auto-submit, then `pool submit-worker --tag %s`",
            config.tag,
        )
    for c in passed:
        do_submit = (
            config.auto_submit
            and getattr(config, "legacy_unsafe_auto_submit", False)
            and c.sim_result and c.sim_result.alpha_id
        )
        if do_submit:
            try:
                sess.submit_alpha(c.sim_result.alpha_id)
                c.submitted = True
                submitted_count += 1
            except Exception as exc:
                logger.warning("Submit failed for %s: %s", c.sim_result.alpha_id, exc)
        pool.add_from_candidate(c, tag=config.tag)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "candidates": len(unique),
        "simulated": len(candidates),
        "passed": len(passed),
        "submitted": submitted_count,
        "pool_size": len(pool),
        # Codex review R3: surface the auto-submit gate state in the JSON
        # summary so operators can distinguish "no candidates passed" from
        # "submit step gated by --legacy-unsafe missing".
        "auto_submit": bool(config.auto_submit),
        "legacy_unsafe_auto_submit": bool(getattr(config, "legacy_unsafe_auto_submit", False)),
        "auto_submit_refused": bool(auto_submit_refused),
        "next_command": (
            f"pool submit-worker --tag {config.tag} --max 20 --one-per-cluster"
            if auto_submit_refused else None
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

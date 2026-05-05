"""Agentic runner — spawn an LLM CLI (opencode/hermes) for one autonomous WQ research session.

The LLM drives the entire workflow: research arxiv abstracts, propose
expressions, validate locally, simulate via wq_brain CLI, learn from results,
iterate, submit. We supply the system prompt + working dir + sandboxed agent
home; the LLM does the rest.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .crossover import extract_top_segments, format_crossover_block, infer_family
from .mutation import render_top_failures_block
from .operators import operators_prompt_block
from .paths import alpha_pool_path, repo_root, tried_exprs_path, wq_brain_run_dir
from .pool import AlphaPool
from .scoring import score_record
from .tried_log import format_for_prompt, read_tried

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    tag: str
    region: str = "USA"
    universe: str = "TOP3000"
    decay: int = 6
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    quality_sharpe_min: float = 1.25
    quality_fitness_min: float = 1.0
    auto_submit: bool = False
    max_turns: int = 100
    cli: str = "auto"  # "auto" | "hermes" | "opencode"
    model: str = ""
    provider: str = ""
    yolo: bool = True
    toolsets: str = "terminal,file"
    reasoning_effort: str = ""
    timeout_sec: float = 7200.0


def _make_run_id(tag: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"wqbrain_agent_{tag}_{ts}_{uuid.uuid4().hex[:8]}"


def _load_dotenv_into(env: dict[str, str], dotenv: Path) -> None:
    if not dotenv.exists():
        return
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        env.setdefault(k.strip(), v)


def _llm_cli_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = os.environ.copy()
    _load_dotenv_into(env, repo_root() / ".env")
    base = env.get("OPENAI_BASE_URL") or env.get("LLM_BASE_URL") or ""
    if base and not base.rstrip("/").endswith("/v1"):
        env["OPENAI_BASE_URL"] = base.rstrip("/") + "/v1"
    if not env.get("LLM_BASE_URL") and env.get("OPENAI_BASE_URL"):
        env["LLM_BASE_URL"] = env["OPENAI_BASE_URL"]
    if not env.get("LLM_API_KEY") and env.get("OPENAI_API_KEY"):
        env["LLM_API_KEY"] = env["OPENAI_API_KEY"]
    no_proxy = env.get("NO_PROXY", "")
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in no_proxy:
            no_proxy = host if not no_proxy else f"{no_proxy},{host}"
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    # Ensure ~/.local/bin is on PATH for tmux/non-login shells where
    # opencode/hermes/claude were installed via uv/pipx.
    local_bin = str(Path.home() / ".local" / "bin")
    path = env.get("PATH", "")
    if local_bin not in path.split(":"):
        env["PATH"] = f"{local_bin}:{path}" if path else local_bin
    # opencode reads provider/model registry from $OPENCODE_CONFIG (project-level)
    project_config = repo_root() / ".opencode.json"
    if project_config.exists():
        env.setdefault("OPENCODE_CONFIG", str(project_config))
    if extra:
        env.update(extra)
    return env


def _prepare_hermes_home(run_dir: Path, env: dict[str, str]) -> Path:
    source = Path(env.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()
    target = run_dir / "hermes_home"
    target.mkdir(parents=True, exist_ok=True)
    for fname in ("config.yaml", "auth.json"):
        src = source / fname
        dst = target / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            if fname == "auth.json":
                try:
                    dst.chmod(0o600)
                except OSError:
                    pass
    src_env = source / ".env"
    if src_env.exists() and source != target:
        _load_dotenv_into(env, src_env)
    env["HERMES_HOME"] = str(target)
    return target


def _resolve_cli(requested: str, *, env: Optional[dict[str, str]] = None) -> str:
    if requested != "auto":
        return requested
    path = (env or os.environ).get("PATH", os.environ.get("PATH", ""))
    if shutil.which("opencode", path=path):
        return "opencode"
    if shutil.which("hermes", path=path):
        return "hermes"
    raise RuntimeError("No agentic CLI found on PATH; install opencode or hermes")


# Full canonical family roster — agent_brief lists these 8 as acceptable;
# crossover.infer_family adds finer subdivisions but the prompt-level
# rotation policy uses this top-level list.
_CANONICAL_FAMILIES: tuple[str, ...] = (
    "ts_corr_pv", "intraday_range", "vwap_dev", "volume_rank",
    "open_gap", "humped", "multi_signal", "sector_relative",
)


def _family_diversity_hint(active_families: list[str]) -> str:
    """Render the 'next candidate must come from family X' hint based on
    what's already in ACTIVE pool. Returns empty string when pool is fresh."""
    if not active_families:
        return ""
    from collections import Counter
    counts = Counter(active_families)
    seen = set(counts)
    missing = [f for f in _CANONICAL_FAMILIES if f not in seen]
    dominant = ", ".join(f"`{name}` (×{n})" for name, n in counts.most_common())
    if missing:
        miss_str = ", ".join(f"`{f}`" for f in missing)
        return (
            f"_⚠️ Pool currently dominated by {dominant}. Your NEXT candidate MUST "
            f"come from a family NOT yet present: {miss_str}._"
        )
    return f"_⚠️ Pool covers {dominant}. Pick the family with the LOWEST count next._"


# Family-specific anti-recommendations — what to try when stuck
_FAMILY_ANTI_EXAMPLES: dict[str, str] = {
    "sector_relative":  "`hump(rank(vwap/close))`, `rank((high - low)/close)`, `rank(open - ts_delay(close, 1))`",
    "ts_corr_pv":       "`rank((vwap - close)/close)`, `hump(rank(...))`, `group_zscore(_, subindustry)`",
    "multi_signal":     "`ts_corr(close, volume, 20)`, `rank((high - low)/close)`, `humped(rank(...))`",
    "ts_rank_close":    "`rank((vwap - close)/close)`, `rank((high - low)/close)`, `rank(open - ts_delay(close, 1))`",
    "ts_delta_close":   "`ts_corr(_, _, _)`, `(high - low) / close`, `group_zscore(_, sector)`",
    "decay_linear":     "swap to a different inner shape — `hump(rank(...))`, `(high - low) / close`, `vwap - close`",
    "humped":           "`ts_corr(close, volume, 20)`, `(high - low)/close`, `group_zscore(_, sector)`",
    "intraday_range":   "`rank(vwap/close)`, `ts_corr(close, volume, 20)`, `group_zscore(_, sector)`",
    "open_gap":         "`rank(vwap/close)`, `(high - low)/close`, `group_zscore(_, sector)`",
    "vwap_dev":         "`rank(ts_rank(volume, 20))`, `rank((high - low)/close)`, `ts_corr(close, volume, 20)`",
    "volume_rank":      "`rank(vwap/close)`, `(high - low)/close`, `hump(rank(...))`",
    "group_neutral":    "`hump(rank(...))`, `ts_corr(close, volume, 20)`, `(high - low)/close`",
    "ts_corr_other":    "`group_zscore(_, sector)`, `(high - low)/close`, `hump(rank(...))`",
}


def _tried_family_concentration_hint(
    tried_records: list[dict],
    *,
    window: int = 10,
    threshold: float = 0.6,
) -> str:
    """Hard rotation hint: if recent attempts are dominated by one family,
    forbid that family for the next handful of candidates.

    Triggered when ≥ ``threshold`` of the last ``window`` attempts share a
    family. Catches the failure mode where mutation engine endlessly twiddles
    parameters within a single family even after that family is already ACTIVE
    (so the ACTIVE-only diversity hint doesn't fire).
    """
    if len(tried_records) < window:
        return ""
    from collections import Counter
    recent = sorted(tried_records, key=lambda r: r.get("ts", 0), reverse=True)[:window]
    families = [infer_family(r.get("expr") or "") for r in recent if r.get("expr")]
    if not families:
        return ""
    counts = Counter(families)
    top_fam, top_count = counts.most_common(1)[0]
    if top_count / window < threshold:
        return ""
    examples = _FAMILY_ANTI_EXAMPLES.get(
        top_fam, "any of the 8 canonical families OTHER than this one"
    )
    return (
        f"_🛑 STUCK IN `{top_fam}` ({top_count}/{window} of recent attempts). The "
        f"mutation engine is over-twiddling parameters inside this family. "
        f"YOUR NEXT 5 simulate calls MUST come from a DIFFERENT family. "
        f"Try: {examples}. Re-entering `{top_fam}` before 5 cross-family "
        f"simulations will be flagged as a session-rule violation._"
    )


def _build_prior_knowledge_block(tag: str, *, max_pool: int = 20, max_tried: int = 60) -> str:
    """Render cross-loop knowledge: passing pool + recent tried_exprs +
    cross-over candidates + mutation hints + submit rejections."""
    parts: list[str] = []

    pool = AlphaPool(alpha_pool_path(tag))
    if len(pool):
        # Split pool by verified_status: ACTIVE (real submissions) vs REJECTED/UNSUBMITTED
        active = [e for e in pool.entries
                  if getattr(e, "verified_status", "QUEUED") == "ACTIVE"]
        rejected = [e for e in pool.entries
                    if getattr(e, "verified_status", "QUEUED") in ("REJECTED", "UNSUBMITTED")
                    and getattr(e, "rejection_reasons", [])]
        queued = [e for e in pool.entries
                  if e not in active and e not in rejected]

        if active:
            top_active = sorted(active, key=lambda e: -e.fitness)[:max_pool]
            active_families = [infer_family(e.expr or "") for e in top_active]
            hint = _family_diversity_hint(active_families)
            lines = [
                "### ✅ ACTIVE Submitted Alphas (TAG=" + tag + ")",
                "",
                f"_{len(active)} alpha(s) verified ACTIVE on WQ. These earn rewards. Avoid"
                f" submitting near-duplicates — WQ self-correlation will reject._",
            ]
            if hint:
                lines.extend(["", hint])
            lines.extend([
                "",
                "| alpha_id | family | expr | sh | fi | to |",
                "|---|---|---|---|---|---|",
            ])
            for e, fam in zip(top_active, active_families):
                expr = (e.expr or "")[:75].replace("|", "/")
                lines.append(
                    f"| {e.alpha_id} | `{fam}` | `{expr}` | {e.sharpe:.2f} | {e.fitness:.2f} | {e.turnover:.2f} |"
                )
            parts.append("\n".join(lines))

        if rejected:
            recent_rejected = sorted(rejected, key=lambda e: -getattr(e, "verified_at", 0))[:max_pool]
            lines = [
                "### 🚫 SUBMIT FAILURES — DO NOT REPEAT THESE STRUCTURES",
                "",
                f"_{len(rejected)} alpha(s) passed local quality gate (sh≥1.25 fi≥1.0)_"
                f" _BUT WERE REJECTED BY WQ. The most common reason is_"
                f" _SELF_CORRELATION ≥ 0.7 against an ACTIVE alpha. If your new_"
                f" _candidate is similar to any of these, IT WILL ALSO BE REJECTED._",
                "",
                "| alpha_id | family | expr | sh | fi | to | failed_check |",
                "|---|---|---|---|---|---|---|",
            ]
            for e in recent_rejected:
                expr = (e.expr or "")[:65].replace("|", "/")
                fam = infer_family(e.expr or "")
                fails = getattr(e, "rejection_reasons", []) or []
                fail_str = ", ".join(
                    f"{r.get('name')}={r.get('value')}" if isinstance(r, dict) else str(r)
                    for r in fails[:2]
                ) or "?"
                lines.append(
                    f"| {e.alpha_id} | `{fam}` | `{expr}` | {e.sharpe:.2f} | {e.fitness:.2f} | {e.turnover:.2f} | {fail_str} |"
                )
            parts.append("\n".join(lines))

        if queued:
            top_queued = sorted(queued, key=lambda e: -e.fitness)[:max_pool]
            lines = [
                "### ⏳ Submission Pending (verification not yet run)",
                "",
                "| alpha_id | expr | sh | fi | to |",
                "|---|---|---|---|---|",
            ]
            for e in top_queued:
                expr = (e.expr or "")[:80].replace("|", "/")
                lines.append(f"| {e.alpha_id} | `{expr}` | {e.sharpe:.2f} | {e.fitness:.2f} | {e.turnover:.2f} |")
            parts.append("\n".join(lines))

    tried = read_tried(tried_exprs_path(tag), tail=max_tried * 4)
    if tried:
        # Hard rotation: if recent attempts are dominated by one family,
        # forbid it for the next 5 candidates. Goes BEFORE the tried table
        # so the agent reads the rule before scanning the (tempting) history.
        rotation_hint = _tried_family_concentration_hint(tried)
        if rotation_hint:
            parts.append(rotation_hint)

        parts.append("### Recently Attempted Expressions (latest result per expr)\n\n" + format_for_prompt(tried, max_rows=max_tried))

        # Cross-over: top fragments by quick-score, diversified by family
        segments = extract_top_segments(tried, min_score=30, top_n=5, diversify_by_family=True)
        cross_block = format_crossover_block(segments)
        if cross_block:
            parts.append(cross_block)

        # Mutation hints: top near-misses with diagnoses
        mutation_block = render_top_failures_block(tried, top_n=3)
        if mutation_block:
            parts.append(mutation_block)

    if not parts:
        return "_(no prior loop data — fresh start)_"
    return "\n\n".join(parts)


def _build_system_prompt(config: AgentConfig, run_dir: Path) -> str:
    brief_path = repo_root() / "src" / "agent_market" / "wq_brain" / "prompts" / "agent_brief.md"
    if not brief_path.exists():
        raise FileNotFoundError(f"Agent brief template not found: {brief_path}")
    template = brief_path.read_text(encoding="utf-8")
    op_block = operators_prompt_block()
    prior_block = _build_prior_knowledge_block(config.tag)
    return template.format(
        TAG=config.tag,
        REGION=config.region,
        UNIVERSE=config.universe,
        DECAY=config.decay,
        NEUTRALIZATION=config.neutralization,
        SHARPE_MIN=config.quality_sharpe_min,
        FITNESS_MIN=config.quality_fitness_min,
        MAX_TURNS=config.max_turns,
        AUTO_SUBMIT="yes" if config.auto_submit else "no",
        OPERATORS=op_block,
        PRIOR_KNOWLEDGE=prior_block,
        REPO_ROOT=str(repo_root()),
        RUN_DIR=str(run_dir),
        WQ_TOOLS=str(repo_root() / "scripts" / "wq_brain.py"),
    )


def _build_hermes_cmd(config: AgentConfig, prompt: str, run_dir: Path, env: dict[str, str]) -> list[str]:
    _prepare_hermes_home(run_dir, env)
    if config.reasoning_effort:
        cfg_proc = subprocess.run(
            ["hermes", "config", "set", "agent.reasoning_effort", config.reasoning_effort.lower()],
            cwd=str(run_dir), env=env, text=True, capture_output=True, timeout=60.0, check=False,
        )
        if cfg_proc.returncode != 0:
            logger.warning("hermes config set failed: %s", cfg_proc.stdout or cfg_proc.stderr)

    cmd = [
        "hermes", "chat", "-Q",
        "--toolsets", config.toolsets,
        "--max-turns", str(config.max_turns),
        "--source", "wq-brain-agent",
    ]
    if config.model:
        cmd.extend(["-m", config.model])
    if config.provider:
        cmd.extend(["--provider", config.provider])
    if config.yolo:
        cmd.append("--yolo")
    cmd.extend(["-q", prompt])
    return cmd


def _build_opencode_cmd(config: AgentConfig, prompt: str) -> list[str]:
    if not config.model:
        raise RuntimeError("opencode requires --model (or OPENCODE_MODEL env)")
    model = config.model if "/" in config.model else f"custom/{config.model}"
    return ["opencode", "run", "-m", model, prompt]


def _build_iter_review(
    *, tag: str, start_ts: float, end_ts: float,
    sharpe_min: float, fitness_min: float,
) -> dict:
    tried = read_tried(tried_exprs_path(tag), tail=2000)
    iter_tried = [
        t for t in tried
        if start_ts <= float(t.get("ts") or 0) <= end_ts + 1.0
    ]
    simulated = len(iter_tried)
    completed = [t for t in iter_tried if t.get("status") == "COMPLETE"]
    passed = [
        t for t in completed
        if (t.get("sharpe") is not None and float(t["sharpe"]) >= sharpe_min)
        and (t.get("fitness") is not None and float(t["fitness"]) >= fitness_min)
    ]
    errored = [t for t in iter_tried if t.get("status") in ("ERROR", "FAILED")]

    def _fi(t: dict) -> float:
        v = t.get("fitness")
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    top3 = sorted(completed, key=_fi, reverse=True)[:3]

    # Multi-dim score for every iter candidate
    grades = {"A": 0, "B": 0, "C": 0, "D": 0}
    scored_top3: list[dict] = []
    for t in top3:
        try:
            s = score_record(t)
            grades[s.grade] = grades.get(s.grade, 0) + 1
            scored_top3.append({
                "expr": t.get("expr"), "sh": t.get("sharpe"),
                "fi": t.get("fitness"), "to": t.get("turnover"),
                "alpha_id": t.get("alpha_id"),
                "score": round(s.score, 1),
                "grade": s.grade,
                "recommendation": s.recommendation,
            })
        except Exception:
            scored_top3.append({
                "expr": t.get("expr"), "sh": t.get("sharpe"),
                "fi": t.get("fitness"), "to": t.get("turnover"),
                "alpha_id": t.get("alpha_id"),
            })
    # Grade distribution across all completed (not just top 3)
    for t in completed:
        if t in top3:
            continue
        try:
            s = score_record(t)
            grades[s.grade] = grades.get(s.grade, 0) + 1
        except Exception:
            pass

    pool = AlphaPool(alpha_pool_path(tag))
    return {
        "tag": tag,
        "iter_window": {"start_ts": start_ts, "end_ts": end_ts,
                        "duration_sec": round(end_ts - start_ts, 1)},
        "iter_simulated": simulated,
        "iter_completed": len(completed),
        "iter_passed": len(passed),
        "iter_errored": len(errored),
        "grade_distribution": grades,
        "top_3_by_fitness": scored_top3,
        "passed_alpha_ids": [t.get("alpha_id") for t in passed if t.get("alpha_id")],
        "pool_size_after": len(pool),
        "thresholds": {"sharpe_min": sharpe_min, "fitness_min": fitness_min},
    }


def _format_review_oneline(review: dict) -> str:
    top = review.get("top_3_by_fitness") or []
    top_fi = top[0].get("fi") if top and top[0].get("fi") is not None else "-"
    top_id = top[0].get("alpha_id") if top else "-"
    top_grade = top[0].get("grade", "-") if top else "-"
    grades = review.get("grade_distribution", {})
    grade_str = "/".join(f"{g}:{grades.get(g, 0)}" for g in ("A", "B", "C", "D"))
    return (
        f"review[tag={review['tag']}] "
        f"sim={review['iter_simulated']} "
        f"complete={review['iter_completed']} "
        f"passed={review['iter_passed']} "
        f"err={review['iter_errored']} "
        f"top_fi={top_fi} "
        f"top_grade={top_grade} "
        f"grades={grade_str} "
        f"top_id={top_id} "
        f"pool={review['pool_size_after']}"
    )


def run_agent(config: AgentConfig) -> dict:
    run_id = _make_run_id(config.tag)
    run_dir = wq_brain_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _llm_cli_env({"WQB_TAG": config.tag, "WQB_RUN_DIR": str(run_dir)})
    cli = _resolve_cli(config.cli, env=env)
    prompt = _build_system_prompt(config, run_dir)
    (run_dir / "system_prompt.md").write_text(prompt, encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "_resolved_cli": cli}, indent=2, default=str),
        encoding="utf-8",
    )

    if not config.model and env.get("LLM_MODEL"):
        config = AgentConfig(**{**asdict(config), "model": env["LLM_MODEL"]})

    if cli == "hermes":
        cmd = _build_hermes_cmd(config, prompt, run_dir, env)
    elif cli == "opencode":
        cmd = _build_opencode_cmd(config, prompt)
    else:
        raise ValueError(f"unknown cli: {cli}")

    log_path = run_dir / "agent.log"
    logger.info("Starting %s agent in %s (max_turns=%d, timeout=%.0fs)",
                cli, run_dir, config.max_turns, config.timeout_sec)
    start = time.time()
    rc: int
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(run_dir),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=config.timeout_sec,
                check=False,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        logger.warning("Agent session timed out after %.0fs", config.timeout_sec)

    end = time.time()
    elapsed = end - start

    review = _build_iter_review(
        tag=config.tag, start_ts=start, end_ts=end,
        sharpe_min=config.quality_sharpe_min,
        fitness_min=config.quality_fitness_min,
    )
    (run_dir / "iter_review.json").write_text(
        json.dumps(review, indent=2, default=str), encoding="utf-8"
    )
    one_line = _format_review_oneline(review)
    logger.info(one_line)

    summary: dict[str, object] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "cli": cli,
        "elapsed_sec": elapsed,
        "agent_returncode": rc,
        "log_path": str(log_path),
        "review": review,
    }
    for fname in ("notes.md", "summary.md", "pool.json"):
        if (run_dir / fname).exists():
            summary[f"agent_{fname}"] = True

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

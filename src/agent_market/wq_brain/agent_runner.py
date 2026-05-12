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
import re
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .operators import operators_prompt_block
from .paths import alpha_pool_path, repo_root, tried_exprs_path, wq_brain_run_dir
from .pool import AlphaPool
from .prompt_builder import (
    _CANONICAL_FAMILIES,
    _FAMILY_ANTI_EXAMPLES,
    _SKELETON_OP_PATTERN,
    _build_prior_knowledge_block,
    _family_diversity_hint,
    _operator_skeleton,
    _tried_family_concentration_hint,
    _tried_skeleton_concentration_hint,
)
from .scoring import score_record
from .tried_log import read_checkpoint, read_tried, write_checkpoint

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
    # Normalize the 3 documented base-URL aliases (README mentions
    # OPENAI_API_BASE in §195/209; OPENAI_BASE_URL in §46; LLM_BASE_URL is
    # the legacy hermes name). First non-empty wins.
    base = (
        env.get("OPENAI_BASE_URL")
        or env.get("OPENAI_API_BASE")
        or env.get("LLM_BASE_URL")
        or ""
    )
    if base and not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    if base:
        # Force-overwrite all three aliases so downstream LLM CLIs see the
        # SAME normalized URL regardless of which alias the user set.
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        env["LLM_BASE_URL"] = base
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
    return [
        "opencode", "run",
        "--print-logs",
        "--log-level", "ERROR",
        "-m", model,
        prompt,
    ]


# Patterns the agent CLI / LLM provider emit on common failure modes. Each
# tuple: (regex, kind, hint). First match wins; "unknown" is the fallback.
_FAILURE_PATTERNS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(r"\[agent_runner\] timeout after|process group ignored SIGTERM",
                re.IGNORECASE),
     "timeout",
     "Agent session reached timeout_sec and the runner terminated its process "
     "group. Reduce per-iteration scope or increase --timeout-sec if a full "
     "agent cycle needs more wall time."),
    (re.compile(r"(剩余额度|用户.{0,4}额度|账号.*额度|额度不足|"
                r"insufficient balance|out of credit|quota[_ ]exhaust|"
                r"rate.?limit|usage limit)", re.IGNORECASE),
     "llm_quota",
     "LLM provider returned a quota/credit-exhausted error. Top up the account "
     "or switch to a different OPENAI_BASE_URL before re-running."),
    (re.compile(r"(401|403|unauthorized|forbidden|session.*expir|"
                r"login required|wq.*auth|invalid.*token)", re.IGNORECASE),
     "wq_auth",
     "WQ session expired or credentials were rejected (401/403). Re-run "
     "`wq_brain auth` and confirm the .env WQ_EMAIL / WQ_PASSWORD values."),
    (re.compile(r"(connection refused|connection reset|timed out|"
                r"name resolution|dns lookup failed|network is unreachable)",
                re.IGNORECASE),
     "network",
     "Network failure (connection refused / DNS / timeout). Check the "
     "machine's outbound to api.worldquantbrain.com + the LLM endpoint."),
    (re.compile(r"(killed|signal 9|sigkill|terminated by signal)", re.IGNORECASE),
     "killed",
     "Agent CLI was killed by the OS (likely OOM or tmux pane close). The "
     "loop can resume from the next iteration; check dmesg for OOM."),
    (re.compile(r"(model not found|model_not_found|no available channel|"
                r"providermodelnotfound|invalid api[ _]key|api[ _]key)",
                re.IGNORECASE),
     "llm_config",
     "LLM CLI received an invalid API key / unavailable model. Verify "
     "OPENAI_API_KEY, OPENAI_MODEL, and the .opencode.json model registry."),
)


def _classify_agent_failure(log_path: Path, *, tail_bytes: int = 4000) -> dict:
    """Read the last few KB of the agent log and tag a failure kind.

    Returns ``{"kind": "...", "hint": "...", "tail": "<truncated log>"}``.
    """
    if not log_path.exists():
        return {"kind": "unknown", "hint": "no log file written", "tail": ""}
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
            tail = f.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return {"kind": "unknown", "hint": f"could not read log: {exc}", "tail": ""}
    for pat, kind, hint in _FAILURE_PATTERNS:
        if pat.search(tail):
            return {"kind": kind, "hint": hint, "tail": tail[-1500:]}
    return {
        "kind": "unknown",
        "hint": "Agent exited non-zero with no recognised failure pattern; "
                "inspect the full log_path for clues.",
        "tail": tail[-1500:],
    }


def _run_cli_with_group_timeout(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_sec: float,
) -> int:
    """Run an agent CLI and kill its whole process group on timeout.

    Agent CLIs spawn shell tools; killing only the direct opencode/hermes
    process can leave orphan WQ simulations burning quota in the background.
    """
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            logf.write(f"\n[agent_runner] timeout after {timeout_sec:.0f}s; terminating process group\n")
            logf.flush()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                logf.write("[agent_runner] process group ignored SIGTERM; sending SIGKILL\n")
                logf.flush()
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=15.0)
            return -1


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
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("score_record failed for top-3 entry: %s", exc)
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
        except (KeyError, TypeError, ValueError):
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

    extra_env = {"WQB_TAG": config.tag, "WQB_RUN_DIR": str(run_dir)}
    if config.max_turns <= 20:
        # Compact autonomous loops should finish and hand off quickly. The
        # agent prompt asks for 1-2 local screens, but LLM CLIs can launch
        # tool calls concurrently; enforce the same budget in the child CLI.
        extra_env["WQB_AGENT_LOCAL_SIM_LIMIT"] = "2"
        extra_env["WQB_AGENT_LOCAL_SIM_MAX_CONCURRENT"] = "1"
        extra_env["WQB_AGENT_LOCAL_SIM_TIMEOUT_SEC"] = "360"
        extra_env["WQB_AGENT_REQUIRE_LOCAL_SIM"] = "1"
    env = _llm_cli_env(extra_env)
    cli = _resolve_cli(config.cli, env=env)
    prompt = _build_system_prompt(config, run_dir)
    (run_dir / "system_prompt.md").write_text(prompt, encoding="utf-8")

    # Resolve model BEFORE writing config.json so the persisted record
    # reflects what actually ran. Order:
    #   1. OPENCODE_MODEL — referenced by ``_build_opencode_cmd`` error msg
    #   2. LLM_MODEL — legacy hermes name
    #   3. OPENAI_MODEL — README §48 / OpenAI-compatible providers
    if not config.model:
        for key in ("OPENCODE_MODEL", "LLM_MODEL", "OPENAI_MODEL"):
            v = env.get(key)
            if v:
                config = AgentConfig(**{**asdict(config), "model": v})
                break

    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "_resolved_cli": cli}, indent=2, default=str),
        encoding="utf-8",
    )

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
    rc = _run_cli_with_group_timeout(
        cmd,
        cwd=run_dir,
        env=env,
        log_path=log_path,
        timeout_sec=config.timeout_sec,
    )
    if rc == -1:
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
    if rc != 0:
        summary["failure"] = _classify_agent_failure(log_path)
        logger.warning(
            "Agent exited rc=%d kind=%s — %s",
            rc,
            summary["failure"].get("kind"),
            summary["failure"].get("hint"),
        )
    for fname in ("notes.md", "summary.md", "pool.json"):
        if (run_dir / fname).exists():
            summary[f"agent_{fname}"] = True

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Persist a checkpoint sidecar next to tried_exprs.jsonl. Lets an
    # outer tmux loop, a watchdog, or a future resume tool know which
    # iteration finished successfully.
    try:
        write_checkpoint(
            tried_exprs_path(config.tag),
            session_id=run_id,
            last_iter=int(review.get("iter_simulated") or 0),
            extra={
                "run_dir": str(run_dir),
                "rc": rc,
                "elapsed_sec": round(elapsed, 1),
                "tag": config.tag,
                "failure_kind": summary.get("failure", {}).get("kind") if rc != 0 else None,
            },
        )
    except OSError as exc:
        logger.warning("failed to write checkpoint sidecar: %s", exc)
    return summary

"""WQBrainRunner — main iteration loop for WorldQuant BRAIN alpha mining.

Architecture mirrors factor_lab.strategy_loop.StrategyLoopRunner but is
completely independent: no dependency on freqtrade / rank_portfolio / LEAN.

LLM invocation reuses the hermes CLI subprocess pattern from strategy_loop.py
(_hermes_cli_env, _prepare_hermes_run_home, _hermes_model, _run_hermes_cli).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent_market import paths as repo_paths

from .client import WQSession, session_from_env
from .dtypes import (
    AlphaCandidate,
    AlphaSettings,
    Phase,
    SimulationResult,
    WQBrainConfig,
    WQBrainState,
)
from .knowledge_base import KnowledgeBase
from .mutation import generate_mutations
from .operators import validate_expression
from .paths import alpha_pool_path, hermes_home_dir, kb_index_path, wq_brain_run_dir
from .pool import AlphaPool
from .prompts import (
    PROMPT_VERSION,
    build_alpha_gen_prompt,
    build_repair_prompt,
    score_history_to_summary,
)
from .registry import append_run_meta
from .session_log import append_round, write_final_summary, write_session_metadata

logger = logging.getLogger(__name__)

# ── run_id helpers ─────────────────────────────────────────────────────────


def make_wqb_run_id(tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._") or "wqb"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"wqbrain_{safe}_{stamp}_{uuid.uuid4().hex[:8]}"


# ── Atomic checkpoint ───────────────────────────────────────────────────────


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.json"


def _save_checkpoint(state: WQBrainState, run_dir: Path) -> None:
    cp = _checkpoint_path(run_dir)
    cp.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    tmp = cp.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.rename(cp)
    try:
        dir_fd = os.open(str(cp.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _load_checkpoint(run_dir: Path) -> Optional[WQBrainState]:
    cp = _checkpoint_path(run_dir)
    tmp = cp.with_suffix(".tmp")
    # Prefer .tmp (interrupted atomic write = more recent) over .json
    for path in (tmp, cp):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if path == tmp:
                    path.rename(cp)  # promote to canonical path
                return WQBrainState.from_dict(data)
            except Exception as exc:
                logger.warning("Checkpoint read failed (%s): %s", path, exc)
    return None


# ── Hermes utilities (copied from strategy_loop.py to avoid import coupling) ─


def _load_dotenv_into(env: dict[str, str], path: Optional[Path] = None) -> None:
    env_path = path or repo_paths.REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        raw = env_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line in raw.splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.lower().startswith("export "):
            item = item[7:].strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env.setdefault(key, value)


def _hermes_cli_env(
    base_env: Optional[Mapping[str, str]] = None, *, load_dotenv: bool = True
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if load_dotenv:
        _load_dotenv_into(env)
    if not str(env.get("OPENAI_BASE_URL") or "").strip():
        api_base = str(env.get("OPENAI_API_BASE") or env.get("LLM_BASE_URL") or "").strip().rstrip("/")
        if api_base:
            env["OPENAI_BASE_URL"] = api_base if api_base.endswith("/v1") else f"{api_base}/v1"
    if not str(env.get("OPENAI_API_KEY") or "").strip() and str(env.get("LLM_API_KEY") or "").strip():
        env["OPENAI_API_KEY"] = str(env.get("LLM_API_KEY") or "")
    no_proxy_raw = env.get("NO_PROXY") or env.get("no_proxy") or ""
    no_proxy_vals = [x.strip() for x in no_proxy_raw.split(",") if x.strip()]
    for extra in ("127.0.0.1", "localhost", "::1"):
        if extra not in no_proxy_vals:
            no_proxy_vals.append(extra)
    no_proxy = ",".join(no_proxy_vals)
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env


def _prepare_hermes_run_home(run_id: str, env: dict[str, str]) -> Path:
    source_home = Path(str(env.get("HERMES_HOME") or Path.home() / ".hermes")).expanduser()
    hermes_home = hermes_home_dir(run_id)
    hermes_home.mkdir(parents=True, exist_ok=True)
    for fname in ("config.yaml", "auth.json"):
        src = source_home / fname
        dst = hermes_home / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            if fname == "auth.json":
                try:
                    dst.chmod(0o600)
                except OSError:
                    pass
    src_env = source_home / ".env"
    if src_env.exists() and source_home != hermes_home:
        _load_dotenv_into(env, src_env)
    env["HERMES_HOME"] = str(hermes_home)
    return hermes_home


def _hermes_model(config_model: str, env: Mapping[str, str]) -> str:
    for value in (
        config_model,
        env.get("HERMES_MODEL", ""),
        env.get("LLM_MODEL", ""),
        env.get("OPENAI_MODEL", ""),
    ):
        raw = str(value or "").strip()
        if raw:
            return raw.split("/", 1)[1] if raw.startswith("custom/") else raw
    return ""


_HERMES_PROBE_PATHS = [
    "/mnt/SSD_4TB/zechuan/codex_tools/hermes_home/hermes-agent/venv-linux/bin",
    "/mnt/SSD_4TB/zechuan/codex_tools/hermes_home/hermes-agent/venv/bin",
    "/mnt/SSD_4TB/zechuan/codex_tools/hermes_home/hermes-agent",
]


def _find_hermes() -> Optional[str]:
    if h := shutil.which("hermes"):
        return h
    for p in _HERMES_PROBE_PATHS:
        candidate = os.path.join(p, "hermes")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _make_wq_claude_md(tried_exprs: Optional[list[dict]] = None) -> str:
    """Persistent WQ skill context written to hermes working directory as CLAUDE.md.

    Hermes (Claude Code) auto-loads CLAUDE.md from its cwd, giving the LLM
    background skill context independent of the per-iteration prompt.
    Includes a rolling "already tried" section so the LLM does not repeat expressions.
    """
    base = """\
# WorldQuant BRAIN Alpha Mining — Persistent Session Context

You are an automated alpha researcher for WorldQuant BRAIN.
Each session: generate FASTEXPR alpha expressions → write `candidate.json` to cwd.

## Output Contract
- Write ONLY `candidate.json`. No other text output.
- Schema: {"candidates": [{"expr": "...", "rationale": "..."}], "prompt_version": "...", "iteration": N}

## Unavailable Operators (will FAIL — never use)
  ts_std, ts_min, ts_product, ts_decay_exp, ts_regression_slope, ts_regression_intercept
  → Use ts_zscore instead of ts_std; ts_mean instead of ts_min

## Unavailable Fields (will FAIL — never use)
  adv60, adv120, adv180, cap
  Fundamental: roe, book_to_price, earnings_yield, pe_ratio, etc.
  → Use adv20 for volume scaling; close/vwap/returns for price signals

## Turnover Constraint (CRITICAL — most failures come from HIGH_TURNOVER)
  Hard limit: turnover > 0.70 → FAIL regardless of sharpe
  Target: turnover < 0.58 for fi ≥ 1.0

  Patterns with HIGH turnover (avoid):
    ts_delta(close, 1) — 1-day delta → to ≈ 0.70
    ts_zscore(returns, 5) — short window → to ≈ 0.70
    -returns — raw daily return → to ≈ 0.72

  Patterns with LOW turnover (prefer):
    ts_mean(returns, 60) — 60-day avg → to ≈ 0.10
    ts_delta(close, 5) / close — 5-day price change → to ≈ 0.25
    ts_rank(close, 252) — annual rank → to ≈ 0.12

## Fitness Formula
  fi ≈ sqrt(|annual_return|) × sharpe / sqrt(turnover)
  Need: fi ≥ 1.0 AND sharpe ≥ 1.25
  With sh=1.8, ret≈18%/yr: need to ≤ 0.58

## Nesting Rules
  BAD (causes timeout): ts_mean(group_zscore(x, sector), 20)
  GOOD: rank(group_zscore(ts_mean(x, 20), sector))  — group OUTSIDE ts
"""
    if not tried_exprs:
        return base

    lines = ["\n## ALREADY SIMULATED — Do NOT reproduce these (exact or near-identical)\n"]
    lines.append("  expr | sharpe | fitness | turnover")
    lines.append("  ---- | ------ | ------- | --------")
    for e in tried_exprs[-30:]:
        sh = f"{e['sh']:.2f}" if e.get("sh") is not None else "timeout"
        fi = f"{e['fi']:.2f}" if e.get("fi") is not None else "timeout"
        to = f"{e['to']:.2f}" if e.get("to") is not None else "N/A"
        lines.append(f"  {e['expr'][:72]} | {sh} | {fi} | {to}")
    lines.append("\nGenerate expressions with DIFFERENT structure, operators, and windows.\n")
    return base + "\n".join(lines)


# Field and operator substitution table for auto-fix purification
_AUTO_FIX_FIELDS: dict[str, str] = {
    "adv60": "adv20",
    "adv120": "adv20",
    "adv180": "adv20",
    "cap": "adv20",
}
_AUTO_FIX_OPS: dict[str, str] = {
    "ts_std": "ts_zscore",
    "ts_min": "ts_mean",
}
_FUNDAMENTAL_FIELDS = frozenset([
    "book_to_price", "earnings_yield", "sales_yield", "ebitda_yield",
    "fcf_yield", "dividend_yield", "debt_to_equity", "debt_to_assets",
    "roe", "roa", "gross_margin", "net_margin", "revenue_growth",
    "earnings_growth", "pe_ratio", "pb_ratio", "ps_ratio",
    "analyst_consensus", "analyst_revision",
])


def _auto_fix_expr(expr: str) -> tuple[str, list[str]]:
    """Apply token-level substitutions for common unavailable fields/operators.

    Returns (fixed_expr, list_of_substitutions_made).
    If no fix is possible (e.g. uses fundamental fields), returns ("", []).
    """
    import re as _re
    # Block if fundamental field is used — no simple substitute
    toks = set(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    if toks & _FUNDAMENTAL_FIELDS:
        return "", [f"fundamental field not fixable: {list(toks & _FUNDAMENTAL_FIELDS)[:2]}"]
    fixed = expr
    fixes: list[str] = []
    for bad, good in {**_AUTO_FIX_FIELDS, **_AUTO_FIX_OPS}.items():
        pattern = r"\b" + _re.escape(bad) + r"\b"
        if _re.search(pattern, fixed):
            fixed = _re.sub(pattern, good, fixed)
            fixes.append(f"{bad}→{good}")
    return fixed, fixes


def _run_hermes_cli(
    idir: Path,
    prompt: str,
    config: WQBrainConfig,
    *,
    env: Optional[Mapping[str, str]] = None,
    tried_exprs: Optional[list[dict]] = None,
) -> None:
    hermes_bin = _find_hermes()
    if hermes_bin is None:
        raise RuntimeError("hermes not found on PATH — install hermes CLI first")
    hermes_env = _hermes_cli_env(env)
    hermes_home = _prepare_hermes_run_home(config.run_id, hermes_env)

    # Write persistent WQ skill context — hermes auto-loads CLAUDE.md from cwd.
    # Always rewrite so the "already tried" section reflects the latest state.
    claude_md_path = idir / "CLAUDE.md"
    claude_md_path.write_text(_make_wq_claude_md(tried_exprs), encoding="utf-8")

    effort = str(
        config.hermes_reasoning_effort or hermes_env.get("HERMES_REASONING_EFFORT") or ""
    ).strip().lower()
    hermes_dir = os.path.dirname(hermes_bin)
    hermes_env = dict(hermes_env)
    hermes_env["PATH"] = hermes_dir + os.pathsep + hermes_env.get("PATH", "")

    if effort:
        cfg_proc = subprocess.run(
            [hermes_bin, "config", "set", "agent.reasoning_effort", effort],
            cwd=str(idir),
            env=hermes_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60.0,
            check=False,
        )
        if cfg_proc.returncode != 0:
            (idir / "agent_response.txt").write_text(cfg_proc.stdout or "", encoding="utf-8")
            raise RuntimeError(
                f"hermes config failed (code {cfg_proc.returncode}); "
                f"see {idir / 'agent_response.txt'}"
            )

    cmd = [
        hermes_bin, "chat", "-Q",
        "--toolsets", str(config.hermes_toolsets or "terminal,file"),
        "--max-turns", str(int(config.max_turns)),
        "--source", "wq-brain",
    ]
    model = _hermes_model(config.model, hermes_env)
    if model:
        cmd.extend(["-m", model])
    provider = str(config.hermes_provider or hermes_env.get("HERMES_PROVIDER") or "").strip()
    if provider:
        cmd.extend(["--provider", provider])
    if config.hermes_yolo:
        cmd.append("--yolo")
    cmd.extend(["-q", prompt])

    proc = subprocess.run(
        cmd,
        cwd=str(idir),
        env=hermes_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(120.0, config.stale_timeout + 300.0),
        check=False,
    )
    prefix = f"Hermes HERMES_HOME={hermes_home}\n" + (f"effort={effort}\n" if effort else "")
    (idir / "agent_response.txt").write_text(prefix + (proc.stdout or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"hermes CLI failed (code {proc.returncode}); "
            f"see {idir / 'agent_response.txt'}"
        )


# ── Candidate JSON parsing ──────────────────────────────────────────────────


def _parse_candidate_json(
    candidate_path: Path, default_settings: AlphaSettings
) -> list[AlphaCandidate]:
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"agent did not write required candidate.json at {candidate_path}"
        )
    data = json.loads(candidate_path.read_text(encoding="utf-8"))
    raw_candidates = data.get("candidates") or []
    if not raw_candidates:
        raise ValueError("candidate.json has empty 'candidates' list")
    result = []
    for item in raw_candidates:
        expr = str(item.get("expr") or "").strip()
        if not expr:
            continue
        result.append(AlphaCandidate(expr=expr, settings=default_settings))
    if not result:
        raise ValueError("No valid expressions found in candidate.json")
    return result


# ── Main runner ─────────────────────────────────────────────────────────────


class WQBrainRunner:
    def __init__(self, config: WQBrainConfig, session: Optional[WQSession] = None) -> None:
        self.config = config
        self._session: Optional[WQSession] = session
        self._run_dir = wq_brain_run_dir(config.run_id)
        self._pool = AlphaPool(alpha_pool_path(config.tag), corr_threshold=config.corr_max)
        self._kb = KnowledgeBase(kb_index_path())

    @property
    def session(self) -> WQSession:
        if self._session is None:
            self._session = session_from_env()
            # Apply CLI --max-concurrent to the session (overrides WQ_MAX_CONCURRENT env var)
            n = self.config.max_concurrent
            if n and n != self._session._max_concurrent:
                import threading as _thr
                self._session._max_concurrent = n
                self._session._global_sem = _thr.Semaphore(n)
            self._session.login()
        return self._session

    def run(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        existing = _load_checkpoint(self._run_dir)
        if existing:
            state = existing
            logger.info(
                "Resuming run %s at iteration %d phase %s",
                self.config.run_id,
                state.iteration,
                state.phase,
            )
        else:
            state = WQBrainState(config=self.config)
            _save_checkpoint(state, self._run_dir)
            write_session_metadata(self._run_dir, self.config.to_dict())

        while state.iteration < self.config.max_iterations:
            state.iteration += 1
            state.phase = Phase.PREPARE
            _save_checkpoint(state, self._run_dir)
            logger.info(
                "[%s] iter=%d/%d starting",
                self.config.run_id,
                state.iteration,
                self.config.max_iterations,
            )
            # Snapshot mutable counters — restored on failure so retry doesn't double-count
            _snap = (
                state.total_generated, state.total_simulated,
                state.total_passed, state.total_submitted,
                len(state.score_history),
            )
            try:
                self._run_iteration(state)
            except Exception as exc:
                logger.error("Iteration %d failed: %s", state.iteration, exc, exc_info=True)
                (state.total_generated, state.total_simulated,
                 state.total_passed, state.total_submitted, hist_len) = _snap
                state.score_history = state.score_history[:hist_len]
                state.iteration -= 1
                state.phase = Phase.PREPARE
                _save_checkpoint(state, self._run_dir)

        append_run_meta(
            self.config.run_id,
            {
                "tag": self.config.tag,
                "total_generated": state.total_generated,
                "total_simulated": state.total_simulated,
                "total_passed": state.total_passed,
                "total_submitted": state.total_submitted,
                "pool_size": len(self._pool),
            },
        )
        self._kb.save()
        write_final_summary(self._run_dir, state.to_dict())
        logger.info("Run %s complete. Pool size: %d", self.config.run_id, len(self._pool))

    def _run_iteration(self, state: WQBrainState) -> None:
        idir = self._run_dir / f"iter_{state.iteration:04d}"
        idir.mkdir(parents=True, exist_ok=True)

        # PREPARE
        state.phase = Phase.PREPARE
        _save_checkpoint(state, self._run_dir)
        self._prepare(idir, state)

        # ALPHA_GEN (or MUTATION every 4th iteration if pool has entries)
        state.phase = Phase.ALPHA_GEN
        _save_checkpoint(state, self._run_dir)
        use_mutation = (
            state.iteration % 4 == 0
            and len(self._pool) >= 3
        )
        if use_mutation:
            candidates = self._mutation_gen(idir, state)
            actual_kb_hits = 0
            logger.info("iter=%d using mutation (%d pool entries)", state.iteration, len(self._pool))
        else:
            candidates, actual_kb_hits = self._alpha_gen(idir, state)
        state.total_generated += len(candidates)

        # SIMULATE
        state.phase = Phase.SIMULATE
        _save_checkpoint(state, self._run_dir)
        candidates = self._simulate(idir, candidates)
        simulated = [c for c in candidates if c.sim_result is not None]
        state.total_simulated += len(simulated)

        # EVALUATE
        state.phase = Phase.EVALUATE
        _save_checkpoint(state, self._run_dir)
        passed = self._evaluate(idir, simulated)
        state.total_passed += len(passed)

        # SUBMIT
        state.phase = Phase.SUBMIT
        _save_checkpoint(state, self._run_dir)
        submitted = self._submit(idir, passed)
        state.total_submitted += len(submitted)

        # ANALYSIS
        state.phase = Phase.ANALYSIS
        _save_checkpoint(state, self._run_dir)
        self._analysis(idir, state, simulated, passed, submitted, used_mutation=use_mutation, kb_hits=actual_kb_hits)

        state.phase = Phase.COMPLETE
        _save_checkpoint(state, self._run_dir)
        logger.info(
            "iter=%d generated=%d simulated=%d passed=%d submitted=%d",
            state.iteration,
            len(candidates),
            len(simulated),
            len(passed),
            len(submitted),
        )

    def _prepare(self, idir: Path, state: WQBrainState) -> None:
        ctx_dir = idir / "context"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        ctx = {
            "iteration": state.iteration,
            "pool_size": len(self._pool),
            "pool_summary": self._pool.summary_for_prompt(),
            "score_history_summary": score_history_to_summary(state.score_history),
            "config": state.config.to_dict(),
        }
        (ctx_dir / "prepare.json").write_text(
            json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _alpha_gen(self, idir: Path, state: WQBrainState) -> list[AlphaCandidate]:
        ctx_path = idir / "context" / "prepare.json"
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))

        # KB retrieval: get relevant examples to inject into prompt
        kb_query = f"{self.config.region} {self.config.universe} low turnover alpha"
        kb_hits = self._kb.search(kb_query, top_k=4)
        kb_examples = [
            f"{e.text[:80]}  # {e.metadata.get('desc', '')} sh={e.metadata.get('sharpe', '?')}"
            for e in kb_hits
        ] if kb_hits else None

        prompt = build_alpha_gen_prompt(
            batch_size=self.config.batch_size,
            iteration=state.iteration,
            pool_summary=ctx["pool_summary"],
            score_history_summary=ctx["score_history_summary"],
            region=self.config.region,
            universe=self.config.universe,
            kb_examples=kb_examples,
        )
        candidate_path = idir / "candidate.json"

        try:
            _run_hermes_cli(idir, prompt, self.config, tried_exprs=state.tried_exprs)
        except RuntimeError as exc:
            if candidate_path.exists():
                logger.warning("hermes exited non-zero but candidate.json found; proceeding. (%s)", exc)
            else:
                raise

        settings = self.config.default_settings()
        raw = _parse_candidate_json(candidate_path, settings)
        return self._purify_candidates(raw, idir), len(kb_hits)

    def _purify_candidates(
        self, candidates: list[AlphaCandidate], idir: Path
    ) -> list[AlphaCandidate]:
        """Auto-purification pass: fix common field/operator errors before simulation.

        Pipeline per candidate:
          1. validate_expression — if clean, pass through
          2. _auto_fix_expr — token-level substitution (ts_std→ts_zscore, adv60→adv20, …)
          3. re-validate — if now clean, replace expr and pass through
          4. still invalid → LLM repair via build_repair_prompt (one hermes call each)
          5. still invalid after LLM — log and drop (save to purification_log.json)
        """
        good: list[AlphaCandidate] = []
        needs_llm_repair: list[tuple[AlphaCandidate, list[str]]] = []

        for c in candidates:
            errors = validate_expression(c.expr)
            if not errors:
                good.append(c)
                continue
            fixed_expr, fixes = _auto_fix_expr(c.expr)
            if fixes and fixed_expr:
                new_errors = validate_expression(fixed_expr)
                if not new_errors:
                    logger.info("Purified (%s): %s → %s", ", ".join(fixes), c.expr[:50], fixed_expr[:50])
                    c.expr = fixed_expr
                    good.append(c)
                    continue
            needs_llm_repair.append((c, errors))

        # LLM repair for expressions that couldn't be auto-fixed
        dropped: list[dict] = []
        for c, errors in needs_llm_repair:
            repaired = self._llm_repair_expr(c, errors[0], idir)
            if repaired is not None:
                good.append(repaired)
            else:
                logger.warning("Dropped invalid expr (no fix): %s | %s", c.expr[:60], errors[0])
                dropped.append({"expr": c.expr, "errors": errors})

        if dropped:
            (idir / "purification_log.json").write_text(
                json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return good

    def _llm_repair_expr(
        self, candidate: AlphaCandidate, error_msg: str, idir: Path
    ) -> Optional[AlphaCandidate]:
        """Use hermes to repair a single invalid expression. Returns fixed candidate or None."""
        repair_dir = idir / f"repair_{candidate.candidate_id}"
        repair_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_repair_prompt(candidate.expr, error_msg)
        repair_path = repair_dir / "candidate.json"
        try:
            _run_hermes_cli(repair_dir, prompt, self.config)
        except RuntimeError as exc:
            if not repair_path.exists():
                logger.debug("LLM repair failed for %s: %s", candidate.expr[:40], exc)
                return None
        try:
            settings = self.config.default_settings()
            repaired = _parse_candidate_json(repair_path, settings)
            if repaired:
                c = repaired[0]
                if not validate_expression(c.expr):
                    logger.info("LLM-repaired: %s → %s", candidate.expr[:40], c.expr[:40])
                    return c
        except Exception as exc:
            logger.debug("Repair parse failed: %s", exc)
        return None

    def _mutation_gen(self, idir: Path, state: WQBrainState) -> list[AlphaCandidate]:
        top_entries = self._pool.top_n_by_fitness(10)
        settings = self.config.default_settings()
        candidates = generate_mutations(top_entries, settings, top_n=5, variants_per_parent=4)
        if not candidates:
            logger.info("Mutation produced 0 candidates, falling back to LLM")
            return self._alpha_gen(idir, state)
        # Write candidate.json so artifacts are consistent
        data = {
            "candidates": [{"expr": c.expr, "rationale": "mutation"} for c in candidates],
            "prompt_version": PROMPT_VERSION,
            "iteration": state.iteration,
        }
        cand_path = idir / "candidate.json"
        (idir / "context").mkdir(parents=True, exist_ok=True)
        self._prepare(idir, state)
        cand_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return candidates

    def _simulate(
        self, idir: Path, candidates: list[AlphaCandidate]
    ) -> list[AlphaCandidate]:
        # Pre-filter: invalid operators and local duplicates
        filtered = []
        for c in candidates:
            errors = validate_expression(c.expr)
            if errors:
                logger.warning("Skipped invalid expr (%s): %s...", errors[0], c.expr[:60])
                c.sim_result = SimulationResult(status="INVALID_EXPR")
                filtered.append(c)
            elif self._pool.is_local_duplicate(c.expr):
                logger.info("Skipped local duplicate: %s...", c.expr[:60])
                c.sim_result = SimulationResult(status="SKIPPED_DUPLICATE")
                filtered.append(c)
            else:
                filtered.append(c)

        to_simulate = [c for c in filtered if c.sim_result is None]
        if to_simulate:
            self.session.batch_simulate(
                to_simulate,
                max_concurrent=self.config.max_concurrent,
                timeout=300.0,
            )

        # Write simulation results to disk
        results_data = [c.to_dict() for c in filtered]
        (idir / "sim_results.json").write_text(
            json.dumps(results_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return filtered

    def _evaluate(
        self, idir: Path, candidates: list[AlphaCandidate]
    ) -> list[AlphaCandidate]:
        passed = []
        for c in candidates:
            r = c.sim_result
            if r is None or not r.check_quality(
                sharpe_min=self.config.quality_sharpe_min,
                fitness_min=self.config.quality_fitness_min,
            ):
                continue
            # Correlation check via WQ API (only for successful simulations with alpha_id)
            if r.alpha_id and not self.config.dry_run:
                try:
                    corr_list = self.session.get_alpha_correlations(r.alpha_id)
                    max_corr = max(
                        (abs(float(x.get("value", 0))) for x in corr_list), default=0.0
                    )
                    c.correlation_ok = max_corr < self.config.corr_max
                    if not c.correlation_ok:
                        logger.info(
                            "Alpha %s rejected: max correlation %.3f >= %.3f",
                            r.alpha_id,
                            max_corr,
                            self.config.corr_max,
                        )
                        continue
                except Exception as exc:
                    logger.warning("Correlation check failed for %s: %s", r.alpha_id, exc)
                    c.correlation_ok = True  # assume ok on error
            else:
                c.correlation_ok = True
            passed.append(c)

        (idir / "passed.json").write_text(
            json.dumps([c.to_dict() for c in passed], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return passed

    def _submit(
        self, idir: Path, passed: list[AlphaCandidate]
    ) -> list[AlphaCandidate]:
        if not self.config.auto_submit or self.config.dry_run:
            if self.config.dry_run:
                logger.info("dry-run: skipping submission of %d alphas", len(passed))
            else:
                logger.info(
                    "--auto-submit not set: skipping WQ pool submission of %d alphas",
                    len(passed),
                )
            return []

        submitted = []
        for c in passed:
            if not c.sim_result or not c.sim_result.alpha_id:
                continue
            try:
                self.session.submit_alpha(c.sim_result.alpha_id)
                c.submitted_to_pool = True
                entry = self._pool.add_from_candidate(c, tag=self.config.tag)
                if entry:
                    submitted.append(c)
                    logger.info(
                        "Submitted alpha %s sharpe=%.2f fitness=%.2f",
                        c.sim_result.alpha_id,
                        c.sim_result.sharpe or 0,
                        c.sim_result.fitness or 0,
                    )
            except Exception as exc:
                logger.warning("Failed to submit alpha %s: %s", c.sim_result.alpha_id, exc)

        (idir / "submitted.json").write_text(
            json.dumps([c.to_dict() for c in submitted], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return submitted

    def _analysis(
        self,
        idir: Path,
        state: WQBrainState,
        simulated: list[AlphaCandidate],
        passed: list[AlphaCandidate],
        submitted: list[AlphaCandidate],
        *,
        used_mutation: bool = False,
        kb_hits: int = 0,
    ) -> None:
        from collections import Counter

        # Aggregate WQ quality-check failures across all completed simulations
        failure_counts: Counter = Counter()
        for c in simulated:
            if c.sim_result and c.sim_result.status == "COMPLETE" and not c.sim_result.check_quality(
                sharpe_min=self.config.quality_sharpe_min,
                fitness_min=self.config.quality_fitness_min,
            ):
                for name in c.sim_result.failed_check_names():
                    failure_counts[name] += 1

        sharpes = [c.sim_result.sharpe for c in simulated if c.sim_result and c.sim_result.sharpe is not None]
        fitnesses = [c.sim_result.fitness for c in simulated if c.sim_result and c.sim_result.fitness is not None]

        # Pick the best expression this iteration (by sharpe among completed)
        best = max(
            (c for c in simulated if c.sim_result and c.sim_result.sharpe is not None),
            key=lambda c: c.sim_result.sharpe,  # type: ignore[union-attr]
            default=None,
        )

        # Record all simulated expressions so the next iteration's CLAUDE.md
        # can show "already tried — do not repeat" to prevent LLM repetition loops.
        for c in simulated:
            sr = c.sim_result
            state.add_tried(
                c.expr,
                sharpe=sr.sharpe if sr else None,
                fitness=sr.fitness if sr else None,
                turnover=sr.turnover if sr else None,
            )

        record: dict[str, Any] = {
            "iteration": state.iteration,
            "generated": len(simulated) + sum(1 for c in simulated if c.sim_result and c.sim_result.status == "SKIPPED_DUPLICATE"),
            "simulated": len(simulated),
            "passed": len(passed),
            "submitted": len(submitted),
            "top_sharpe": best.sim_result.sharpe if best else None,
            "top_fitness": best.sim_result.fitness if best else None,
            "top_expr": best.expr if best else None,
            "failure_summary": dict(failure_counts),
            "pool_size": len(self._pool),
            "used_mutation": used_mutation,
            "kb_hits": kb_hits,
        }
        state.score_history.append(record)

        # Feed passing alphas into KB for future retrieval
        for c in passed:
            if c.sim_result and c.sim_result.alpha_id:
                self._kb.add_alpha(
                    c.expr,
                    sharpe=c.sim_result.sharpe or 0.0,
                    fitness=c.sim_result.fitness or 0.0,
                    turnover=c.sim_result.turnover or 0.0,
                )

        (idir / "evaluation.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        append_round(idir, record)

        logger.info(
            ">>> iter_%04d COMPLETE <<< simulated=%d passed=%d submitted=%d pool=%d failures=%s",
            state.iteration,
            len(simulated),
            len(passed),
            len(submitted),
            len(self._pool),
            dict(failure_counts),
        )

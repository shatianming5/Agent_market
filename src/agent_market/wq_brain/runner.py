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
from .paths import alpha_pool_path, hermes_home_dir, wq_brain_run_dir
from .pool import AlphaPool
from .prompts import (
    PROMPT_VERSION,
    build_alpha_gen_prompt,
    score_history_to_summary,
)
from .registry import append_run_meta

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
    for path in (cp, tmp):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if path == tmp:
                    path.rename(cp)
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


def _run_hermes_cli(
    idir: Path,
    prompt: str,
    config: WQBrainConfig,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    if shutil.which("hermes") is None:
        raise RuntimeError("hermes not found on PATH — install hermes CLI first")
    hermes_env = _hermes_cli_env(env)
    hermes_home = _prepare_hermes_run_home(config.run_id, hermes_env)

    effort = str(
        config.hermes_reasoning_effort or hermes_env.get("HERMES_REASONING_EFFORT") or ""
    ).strip().lower()
    if effort:
        cfg_proc = subprocess.run(
            ["hermes", "config", "set", "agent.reasoning_effort", effort],
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
        "hermes", "chat", "-Q",
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

    @property
    def session(self) -> WQSession:
        if self._session is None:
            self._session = session_from_env()
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
            try:
                self._run_iteration(state)
            except Exception as exc:
                logger.error("Iteration %d failed: %s", state.iteration, exc, exc_info=True)
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
        logger.info("Run %s complete. Pool size: %d", self.config.run_id, len(self._pool))

    def _run_iteration(self, state: WQBrainState) -> None:
        idir = self._run_dir / f"iter_{state.iteration:04d}"
        idir.mkdir(parents=True, exist_ok=True)

        # PREPARE
        state.phase = Phase.PREPARE
        _save_checkpoint(state, self._run_dir)
        self._prepare(idir, state)

        # ALPHA_GEN
        state.phase = Phase.ALPHA_GEN
        _save_checkpoint(state, self._run_dir)
        candidates = self._alpha_gen(idir, state)
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
        self._analysis(idir, state, simulated, passed, submitted)

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

        prompt = build_alpha_gen_prompt(
            batch_size=self.config.batch_size,
            iteration=state.iteration,
            pool_summary=ctx["pool_summary"],
            score_history_summary=ctx["score_history_summary"],
            region=self.config.region,
            universe=self.config.universe,
        )
        candidate_path = idir / "candidate.json"

        _run_hermes_cli(idir, prompt, self.config)

        settings = self.config.default_settings()
        return _parse_candidate_json(candidate_path, settings)

    def _simulate(
        self, idir: Path, candidates: list[AlphaCandidate]
    ) -> list[AlphaCandidate]:
        # Pre-filter obvious local duplicates before hitting the API
        filtered = []
        for c in candidates:
            if self._pool.is_local_duplicate(c.expr):
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
            if r is None or not r.passes_quality:
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
    ) -> None:
        sharpes = [
            c.sim_result.sharpe for c in passed if c.sim_result and c.sim_result.sharpe
        ]
        fitnesses = [
            c.sim_result.fitness for c in passed if c.sim_result and c.sim_result.fitness
        ]
        record: dict[str, Any] = {
            "iteration": state.iteration,
            "simulated": len(simulated),
            "passed": len(passed),
            "submitted": len(submitted),
            "top_sharpe": max(sharpes) if sharpes else None,
            "top_fitness": max(fitnesses) if fitnesses else None,
            "pool_size": len(self._pool),
        }
        state.score_history.append(record)
        (idir / "evaluation.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            ">>> iter_%04d COMPLETE <<< simulated=%d passed=%d submitted=%d pool=%d",
            state.iteration,
            len(simulated),
            len(passed),
            len(submitted),
            len(self._pool),
        )

"""Agentic runner: spawn hermes for one autonomous WQ alpha-research session.

Hermes drives the entire workflow: research arxiv abstracts, propose
expressions, validate locally, simulate via wq_brain CLI, learn from results,
iterate, submit. We supply the system prompt + working dir; hermes does the
rest.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .operators import operators_prompt_block
from .paths import repo_root, wq_brain_run_dir

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
    model: str = ""
    hermes_provider: str = ""
    hermes_yolo: bool = True
    hermes_toolsets: str = "terminal,file"
    hermes_reasoning_effort: str = ""
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


def _hermes_cli_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = os.environ.copy()
    _load_dotenv_into(env, repo_root() / ".env")
    base = env.get("OPENAI_BASE_URL", "")
    if base and not base.rstrip("/").endswith("/v1"):
        env["OPENAI_BASE_URL"] = base.rstrip("/") + "/v1"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    if extra:
        env.update(extra)
    return env


def _build_system_prompt(config: AgentConfig, run_dir: Path) -> str:
    brief_path = repo_root() / "src" / "agent_market" / "wq_brain" / "prompts" / "agent_brief.md"
    if not brief_path.exists():
        raise FileNotFoundError(f"Agent brief template not found: {brief_path}")
    template = brief_path.read_text(encoding="utf-8")
    op_block = operators_prompt_block()
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
        REPO_ROOT=str(repo_root()),
        RUN_DIR=str(run_dir),
        WQ_TOOLS=str(repo_root() / "scripts" / "wq_brain.py"),
    )


def run_agent(config: AgentConfig) -> dict:
    """Spawn hermes for one autonomous session. Returns summary dict."""
    run_id = _make_run_id(config.tag)
    run_dir = wq_brain_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt = _build_system_prompt(config, run_dir)
    (run_dir / "system_prompt.md").write_text(prompt, encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str), encoding="utf-8"
    )

    cmd = [
        "hermes",
        prompt,
        "--toolsets", config.hermes_toolsets,
        "--max-turns", str(config.max_turns),
    ]
    if config.model:
        cmd.extend(["--model", config.model])
    if config.hermes_provider:
        cmd.extend(["--provider", config.hermes_provider])
    if config.hermes_reasoning_effort:
        cmd.extend(["--reasoning-effort", config.hermes_reasoning_effort])
    if config.hermes_yolo:
        cmd.append("--yolo")

    env = _hermes_cli_env({"WQB_TAG": config.tag, "WQB_RUN_DIR": str(run_dir)})

    log_path = run_dir / "hermes.log"
    logger.info("Starting hermes agent in %s (max_turns=%d)", run_dir, config.max_turns)
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
        logger.warning("Hermes session timed out after %.0fs", config.timeout_sec)

    elapsed = time.time() - start

    summary: dict[str, object] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "elapsed_sec": elapsed,
        "hermes_returncode": rc,
        "log_path": str(log_path),
    }
    for fname in ("notes.md", "summary.md", "pool.json"):
        if (run_dir / fname).exists():
            summary[f"agent_{fname}"] = True

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

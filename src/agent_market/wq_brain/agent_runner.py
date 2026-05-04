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

    elapsed = time.time() - start
    summary: dict[str, object] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "cli": cli,
        "elapsed_sec": elapsed,
        "agent_returncode": rc,
        "log_path": str(log_path),
    }
    for fname in ("notes.md", "summary.md", "pool.json"):
        if (run_dir / fname).exists():
            summary[f"agent_{fname}"] = True

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

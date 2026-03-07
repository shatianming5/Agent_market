"""Helpers to build StrategyAgent instances with consistent tool policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_adapter import StrategyAgent
from .dtypes import MinerConfig


def build_tool_policy(config: MinerConfig, workspace: Path) -> Any:
    from runner_fsm.opencode.tool_executor import ToolPolicy  # noqa: WPS433

    allowed = (
        frozenset(
            str(x).strip().lower()
            for x in (config.tool_allowlist or [])
            if str(x).strip()
        )
        if (config.tool_allowlist is not None)
        else None
    )

    return ToolPolicy(
        repo=Path(workspace).resolve(),
        unattended="strict",
        allowed_tool_kinds=allowed,
        bash_allow=bool(config.bash_allow),
        bash_allowlist=tuple(str(x) for x in (config.bash_allowlist or []) if str(x).strip()),
        bash_timeout_seconds=int(config.bash_timeout or 60),
    )


def build_strategy_agent(config: MinerConfig, workspace: Path) -> StrategyAgent:
    tool_policy = build_tool_policy(config, workspace)
    return StrategyAgent(
        workspace=Path(workspace),
        model=config.model,
        base_url=config.base_url,
        max_turns=config.max_turns,
        stale_timeout=config.stale_timeout,
        max_retries=config.max_retries,
        provider=config.provider,
        tool_policy=tool_policy,
    )

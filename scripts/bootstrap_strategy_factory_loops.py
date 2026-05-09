#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOOPS_ROOT = REPO_ROOT / ".codex" / "ralph"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _seed_loop(loop_name: str, prd: str, acceptance: str, focus: str) -> None:
    loop_dir = LOOPS_ROOT / loop_name
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write(loop_dir / "PRD.md", prd)
    _write(loop_dir / "acceptance.md", acceptance)
    _write(
        loop_dir / "progress.md",
        """
# Progress

## Done

- Repo-local Ralph loop scaffold created.

## In Progress

- Start the first bounded implementation iteration.

## Blocked

- None.

## Next

- Read PRD.md and choose one bounded task.
""",
    )
    _write(
        loop_dir / "handoff.md",
        """
# Handoff

- Runtime plane must stay direct OpenCode.
- Ralph loop only manages implementation backlog.
- Update progress.md and acceptance.md after each bounded pass.
""",
    )
    _write(
        loop_dir / "iteration_prompt.md",
        f"""
You are running Ralph loop iteration for `{loop_name}`.

Read PRD.md, acceptance.md, progress.md, and handoff.md.
Choose exactly one bounded task.
Implement it, verify it, then update progress.md, handoff.md, and loop_state.json.
Do not claim completion without an explicit verification result.
Current focus: {focus}
""",
    )
    state = {
        "schema_version": "1.0",
        "loop_name": loop_name,
        "status": "active",
        "iteration": 0,
        "repo_root": str(REPO_ROOT),
        "loop_dir": str(loop_dir),
        "created_at": ts,
        "updated_at": ts,
        "current_focus": focus,
        "last_result": "",
        "next_action": "Read PRD.md and select one bounded task.",
    }
    _write(loop_dir / "loop_state.json", json.dumps(state, ensure_ascii=False, indent=2))


def main() -> int:
    architecture_prd = """
# PRD: OpenCode Strategy Factory Architecture

## Goal

Keep the target architecture in [docs/opencode_strategy_factory_architecture.md](docs/opencode_strategy_factory_architecture.md) aligned with the real repository.

## Scope

- Maintain architecture boundaries between runtime plane and Ralph implementation plane.
- Keep factor mining on direct OpenCode agent execution.
- Keep strategy evolution on direct OpenCode agent execution.
- Keep artifact and budget contracts explicit.

## Bounded backlog examples

- one architecture gap
- one runtime contract fix
- one budget or artifact mismatch
- one doc or config sync fix
"""

    architecture_acceptance = """
# Acceptance

- The architecture doc matches the actual runnable entrypoints in this repo.
- Runtime plane still uses direct OpenCode execution for factor mining and strategy evolution.
- Each iteration updates progress.md, handoff.md, and loop_state.json.
- Verification must include at least one narrow command or file check.
"""

    factor_prd = """
# PRD: Factor Mining Infra Loop

## Goal

Continuously improve the factor-mining runtime so `scripts/freqai_expression_agent.py` and `scripts/agent_flow.py` can use direct OpenCode agents safely and reproducibly.

## Runtime entrypoints

- [scripts/freqai_expression_agent.py](scripts/freqai_expression_agent.py)
- [configs/agent_flow_kucoin_factor_opencode.json](configs/agent_flow_kucoin_factor_opencode.json)

## Bounded backlog examples

- one prompt or JSON extraction hardening fix
- one factor artifact persistence fix
- one evaluation or scoring gap
- one factor card or memory pipeline improvement
"""

    factor_acceptance = """
# Acceptance

- Factor mining still runs through direct OpenCode agent execution.
- The expression CLI accepts provider/workspace/agent settings.
- Generated artifacts remain JSON-parseable and machine-checkable.
- Each iteration records the exact verification command and result.
"""

    strategy_prd = """
# PRD: Strategy Evolution Infra Loop

## Goal

Continuously improve the strategy-miner runtime so grouped alpha configs can be rendered into direct OpenCode runtime configs without breaking existing openai-compatible runs.

## Runtime entrypoints

- [scripts/strategy_miner.py](scripts/strategy_miner.py)
- [scripts/make_strategy_miner_opencode_config.py](scripts/make_strategy_miner_opencode_config.py)

## Bounded backlog examples

- one strategy runtime config gap
- one benchmark or holdout contract gap
- one lineage / knowledge / promotion gap
- one futures discovery efficiency fix
"""

    strategy_acceptance = """
# Acceptance

- Strategy evolution remains direct OpenCode at runtime.
- Runtime config rendering does not mutate base configs.
- Existing grouped alpha configs stay reusable.
- Each Ralph iteration closes exactly one bounded gap with verification.
"""

    _seed_loop("architecture", architecture_prd, architecture_acceptance, "Architecture doc and runtime/control-plane boundary")
    _seed_loop("factor-mining", factor_prd, factor_acceptance, "Direct OpenCode factor-mining runtime")
    _seed_loop("strategy-evolution", strategy_prd, strategy_acceptance, "Direct OpenCode strategy-miner runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

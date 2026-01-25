from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_market import flow_steps

logger = logging.getLogger(__name__)


@dataclass
class AgentFlowConfig:
    feature: Optional[Dict[str, Any]] = None
    expression: Optional[Dict[str, Any]] = None
    ml_training: Optional[Dict[str, Any]] = None
    rl_training: Optional[Dict[str, Any]] = None
    backtest: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentFlowConfig":
        known_keys = {"feature", "expression", "ml_training", "rl_training", "backtest"}
        extra = set(data.keys()) - known_keys
        if extra:
            logger.warning(
                "AgentFlowConfig received unknown keys: %s", ", ".join(sorted(extra))
            )
        return cls(
            feature=data.get("feature"),
            expression=data.get("expression"),
            ml_training=data.get("ml_training"),
            rl_training=data.get("rl_training"),
            backtest=data.get("backtest"),
        )


def load_agent_flow_config(path: Path) -> AgentFlowConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse config JSON: {exc}") from exc
    return AgentFlowConfig.from_dict(payload)


class AgentFlow:
    STEP_ORDER = ["feature", "expression", "ml", "rl", "backtest"]

    def __init__(
        self,
        config: AgentFlowConfig,
        feedback_path: Path | str = "user_data/llm_feedback/latest_backtest_summary.json",
    ):
        self.config = config
        self.feedback_path = Path(feedback_path)

    def run(self, steps: Optional[List[str]] = None) -> None:
        requested: Optional[list[str]] = None
        if steps:
            requested = [str(step).lower() for step in steps]
            unknown = sorted({step for step in requested if step not in self.STEP_ORDER})
            if unknown:
                logger.warning("Ignoring unknown steps: %s", ", ".join(unknown))
            requested = [step for step in requested if step in self.STEP_ORDER]

        sequence: list[tuple[str, Optional[Dict[str, Any]]]] = [
            ("feature", self.config.feature),
            ("expression", self.config.expression),
            ("ml", self.config.ml_training),
            ("rl", self.config.rl_training),
            ("backtest", self.config.backtest),
        ]

        for name, cfg in sequence:
            if requested and name not in requested:
                continue
            if not cfg:
                if requested:
                    logger.warning("Step '%s' requested but no configuration provided", name)
                continue

            logger.info("[FLOW] STEP_START %s", name)
            logger.info("[FLOW] PHASE %s prepare", name)
            try:
                logger.info("[FLOW] PHASE %s execute", name)
                if name == "feature":
                    flow_steps.run_feature_generation(cfg)
                elif name == "expression":
                    flow_steps.run_expression_generation(cfg, self.feedback_path)
                elif name == "ml":
                    flow_steps.run_ml_training(cfg)
                elif name == "rl":
                    flow_steps.run_rl_training(cfg)
                elif name == "backtest":
                    flow_steps.run_backtest(cfg, self.feedback_path)
                else:  # pragma: no cover
                    raise ValueError(f"Unknown step: {name}")
            except Exception as exc:
                logger.error("[FLOW] STEP_FAIL %s: %s", name, exc)
                raise
            else:
                logger.info("[FLOW] PHASE %s summarize", name)
                logger.info("[FLOW] STEP_OK %s", name)


__all__ = ["AgentFlow", "AgentFlowConfig", "load_agent_flow_config"]

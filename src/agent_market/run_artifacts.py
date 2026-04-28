"""Mutable container for tracking artifacts produced during a flow run."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunArtifacts:
    """Tracks output paths produced by each step during AgentFlow.run()."""

    feature_output: Optional[str] = None
    expression_output: Optional[str] = None
    expression_scored_output: Optional[str] = None
    portfolio_weights: Optional[str] = None
    portfolio_report: Optional[str] = None
    portfolio_returns: Optional[str] = None
    micro_feature_parquet: Optional[str] = None
    micro_feature_manifest: Optional[str] = None
    capture_manifest: Optional[str] = None
    capture_match_path: Optional[str] = None
    capture_level2_path: Optional[str] = None
    lob_state_parquet: Optional[str] = None
    rebuild_report: Optional[str] = None
    factor_spec_json: Optional[str] = None
    factor_ast_json: Optional[str] = None
    factor_expression_txt: Optional[str] = None
    factor_expression_json: Optional[str] = None
    factor_eval_meta: Optional[str] = None
    factor_scores_json: Optional[str] = None
    factor_pareto_csv: Optional[str] = None
    factor_memory_json: Optional[str] = None
    factor_cards_json: Optional[str] = None
    factor_failure_cards_json: Optional[str] = None
    factor_lineage_json: Optional[str] = None
    global_factor_memory_json: Optional[str] = None
    global_factor_cards_json: Optional[str] = None
    global_factor_failure_cards_json: Optional[str] = None
    global_factor_lineage_json: Optional[str] = None
    tca_report: Optional[str] = None
    tca_html: Optional[str] = None
    bundle_zip: Optional[str] = None
    bundle_manifest: Optional[str] = None
    strategy_miner_summary: Optional[str] = None
    strategy_miner_dir: Optional[str] = None
    global_strategy_knowledge_base_json: Optional[str] = None
    training_summary_json: Optional[str] = None
    feedback_summary_json: Optional[str] = None
    experiment_registry: Optional[str] = None
    budget_plan_json: Optional[str] = None
    replay_manifest_json: Optional[str] = None
    lineage_graph_json: Optional[str] = None
    promotion_chain_json: Optional[str] = None
    resource_dashboard_json: Optional[str] = None
    backtest_zip: Optional[str] = None
    backtest_zip_run: Optional[str] = None

    # Transient state — not serialised into run_meta.
    compiled_expression_path: Optional[Path] = None
    capture_dir_path: Optional[Path] = None

    def to_dict(self, *, feedback_summary: Optional[str] = None,
                model_dirs: Optional[list[str]] = None,
                training_summaries: Optional[list[str]] = None,
                backtest_results_dir: Optional[str] = None,
                backtest_zips: Optional[list[str]] = None) -> Dict[str, Any]:
        """Return a dict suitable for the ``artifacts`` block in run_meta."""
        return {
            "feature_output": self.feature_output,
            "micro_feature_parquet": self.micro_feature_parquet,
            "micro_feature_manifest": self.micro_feature_manifest,
            "capture_manifest": self.capture_manifest,
            "capture_match_path": self.capture_match_path,
            "capture_level2_path": self.capture_level2_path,
            "lob_state_parquet": self.lob_state_parquet,
            "rebuild_report": self.rebuild_report,
            "portfolio_weights": self.portfolio_weights,
            "portfolio_report": self.portfolio_report,
            "portfolio_returns": self.portfolio_returns,
            "expression_output": self.expression_output,
            "expression_scored_output": self.expression_scored_output,
            "factor_spec_json": self.factor_spec_json,
            "factor_ast_json": self.factor_ast_json,
            "factor_expression_txt": self.factor_expression_txt,
            "factor_expression_json": self.factor_expression_json,
            "factor_eval_meta": self.factor_eval_meta,
            "factor_scores_json": self.factor_scores_json,
            "factor_pareto_csv": self.factor_pareto_csv,
            "factor_memory_json": self.factor_memory_json,
            "factor_cards_json": self.factor_cards_json,
            "factor_failure_cards_json": self.factor_failure_cards_json,
            "factor_lineage_json": self.factor_lineage_json,
            "global_factor_memory_json": self.global_factor_memory_json,
            "global_factor_cards_json": self.global_factor_cards_json,
            "global_factor_failure_cards_json": self.global_factor_failure_cards_json,
            "global_factor_lineage_json": self.global_factor_lineage_json,
            "feedback_summary": feedback_summary,
            "model_dirs": model_dirs or [],
            "training_summaries": training_summaries or [],
            "backtest_results_dir": backtest_results_dir,
            "backtest_zips": backtest_zips or [],
            "tca_report": self.tca_report,
            "tca_html": self.tca_html,
            "bundle_zip": self.bundle_zip,
            "bundle_manifest": self.bundle_manifest,
            "strategy_miner_summary": self.strategy_miner_summary,
            "strategy_miner_dir": self.strategy_miner_dir,
            "global_strategy_knowledge_base_json": self.global_strategy_knowledge_base_json,
            "training_summary_json": self.training_summary_json,
            "feedback_summary_json": self.feedback_summary_json,
            "experiment_registry": self.experiment_registry,
            "budget_plan_json": self.budget_plan_json,
            "replay_manifest_json": self.replay_manifest_json,
            "lineage_graph_json": self.lineage_graph_json,
            "promotion_chain_json": self.promotion_chain_json,
            "resource_dashboard_json": self.resource_dashboard_json,
            "backtest_zip": self.backtest_zip,
            "backtest_zip_run": self.backtest_zip_run,
        }

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
    tca_report: Optional[str] = None
    tca_html: Optional[str] = None
    bundle_zip: Optional[str] = None
    bundle_manifest: Optional[str] = None

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
            "factor_spec_json": self.factor_spec_json,
            "factor_ast_json": self.factor_ast_json,
            "factor_expression_txt": self.factor_expression_txt,
            "factor_expression_json": self.factor_expression_json,
            "factor_eval_meta": self.factor_eval_meta,
            "factor_scores_json": self.factor_scores_json,
            "factor_pareto_csv": self.factor_pareto_csv,
            "feedback_summary": feedback_summary,
            "model_dirs": model_dirs or [],
            "training_summaries": training_summaries or [],
            "backtest_results_dir": backtest_results_dir,
            "backtest_zips": backtest_zips or [],
            "tca_report": self.tca_report,
            "tca_html": self.tca_html,
            "bundle_zip": self.bundle_zip,
            "bundle_manifest": self.bundle_manifest,
        }

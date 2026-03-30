"""Individual step handlers dispatched from AgentFlow.run().

Each handler receives ``(cfg, arts, ctx)`` and mutates *arts* (RunArtifacts)
with the paths it produces.  *ctx* carries shared state such as the run_dir,
feedback_path, and references to other config sections.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market import paths
from agent_market.flow_ext import steps as flow_steps
from agent_market.run_artifacts import RunArtifacts

logger = logging.getLogger(__name__)
REPO_ROOT = paths.REPO_ROOT


def _relpath(path: Path) -> str:
    return paths.relpath_under_repo(path if path.is_absolute() else (REPO_ROOT / path))


@dataclass
class StepContext:
    """Shared state passed to every step handler."""
    run_id: str
    run_dir: Path
    feedback_path: Path
    full_config: Any  # AgentFlowConfig
    config_path: Optional[Path] = None  # original config file path


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------

def _step_feature(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    flow_steps.run_feature_generation(cfg)


def _step_capture(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    out = flow_steps.run_capture(cfg, out_dir=(ctx.run_dir / "capture").resolve())
    arts.capture_dir_path = Path(out["capture_dir"]).resolve()
    arts.capture_manifest = _relpath(Path(out["manifest_json"]))
    arts.capture_match_path = _relpath(Path(out["match_path"]))
    arts.capture_level2_path = _relpath(Path(out["level2_path"]))


def _step_lob_rebuild(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    cap_dir_cfg = cfg.get("capture_dir") if isinstance(cfg, dict) else None
    cap_dir = paths.resolve_repo_path(str(cap_dir_cfg)) if cap_dir_cfg else arts.capture_dir_path
    if cap_dir is None:
        raise ValueError("lob_rebuild requires capture_dir (or run capture step first)")
    out = flow_steps.run_lob_rebuild(cfg, capture_dir=cap_dir, out_dir=(ctx.run_dir / "lob_rebuild").resolve())
    arts.lob_state_parquet = _relpath(Path(out["lob_state_parquet"]))
    arts.rebuild_report = _relpath(Path(out["rebuild_report_json"]))


def _step_micro_feature(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    mf_cfg = dict(cfg)
    fc = ctx.full_config
    if str(mf_cfg.get("mode") or "").strip().lower() == "microstructure":
        if not mf_cfg.get("lob_state") and arts.lob_state_parquet:
            mf_cfg["lob_state"] = arts.lob_state_parquet
        if not mf_cfg.get("match") and arts.capture_match_path:
            mf_cfg["match"] = arts.capture_match_path
        if not mf_cfg.get("symbol") and fc.lob_rebuild:
            sym = (fc.lob_rebuild or {}).get("symbol")
            if sym:
                mf_cfg["symbol"] = sym
        if not mf_cfg.get("depth_levels") and fc.lob_rebuild:
            depth = (fc.lob_rebuild or {}).get("depth")
            if depth:
                mf_cfg["depth_levels"] = depth
    elif not mf_cfg.get("config") and fc.backtest:
        mf_cfg["config"] = fc.backtest.get("config")
    out = flow_steps.run_micro_feature(mf_cfg, run_id=ctx.run_id, out_dir=(ctx.run_dir / "micro_feature").resolve())
    arts.micro_feature_parquet = _relpath(Path(out["features_parquet"]))
    arts.micro_feature_manifest = _relpath(Path(out["manifest_json"]))


def _step_portfolio(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    from agent_market.portfolio_opt import compute_returns, load_prices_from_feather, optimize_hrp  # noqa: WPS433

    ex = str(cfg.get("exchange") or "").strip()
    pairs = cfg.get("pairs") or []
    if isinstance(pairs, str):
        pairs = [p for p in pairs.split(",") if p.strip()]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("portfolio.pairs must be a non-empty list")

    timeframe = str(cfg.get("timeframe") or "1h").strip()
    timerange = cfg.get("timerange") or (ctx.full_config.backtest or {}).get("timerange")
    data_dir = cfg.get("data_dir") or "user_data/data"
    returns_kind = str(cfg.get("returns") or "log").strip().lower()

    prices = load_prices_from_feather(
        REPO_ROOT,
        exchange=ex,
        pairs=[str(p) for p in pairs],
        timeframe=timeframe,
        timerange=str(timerange) if timerange else None,
        data_dir=str(paths.resolve_repo_path(str(data_dir))),
    )
    returns = compute_returns(prices, returns_kind)
    result = optimize_hrp(returns)

    out_dir = (ctx.run_dir / "portfolio").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.json"
    report_path = out_dir / "report.json"
    returns_path = out_dir / "returns.parquet"

    weights_path.write_text(json.dumps({"method": "hrp", "weights": result.get("weights") or {}}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        returns.to_parquet(returns_path, index=True)
    except Exception:
        logger.warning("Failed to save portfolio returns to %s", returns_path, exc_info=True)
        returns_path = None  # type: ignore[assignment]

    report = {
        "method": "hrp",
        "weights": result.get("weights") or {},
        "stats": result.get("stats") or {},
        "inputs": {
            "exchange": ex,
            "pairs": [str(p) for p in pairs],
            "timeframe": timeframe,
            "timerange": str(timerange) if timerange else None,
            "returns_kind": returns_kind,
            "data_dir": str(data_dir),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    arts.portfolio_weights = _relpath(weights_path)
    arts.portfolio_report = _relpath(report_path)
    if returns_path is not None:
        arts.portfolio_returns = _relpath(returns_path)


def _step_expression(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    flow_steps.run_expression_generation(cfg, ctx.feedback_path)


def _step_factor_compile(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    out = flow_steps.run_factor_compile(cfg, run_id=ctx.run_id, out_dir=(ctx.run_dir / "factor_compile").resolve())
    arts.factor_spec_json = _relpath(Path(out["factor_spec_json"]))
    arts.factor_ast_json = _relpath(Path(out["factor_ast_json"]))
    arts.factor_expression_txt = _relpath(Path(out["compiled_expression_txt"]))
    arts.factor_expression_json = _relpath(Path(out["compiled_expression_json"]))
    arts.compiled_expression_path = Path(out["compiled_expression_txt"]).resolve()


def _step_factor_eval(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    fe_cfg = dict(cfg)
    if not fe_cfg.get("features_parquet") and arts.micro_feature_parquet:
        fe_cfg["features_parquet"] = arts.micro_feature_parquet
    out = flow_steps.run_factor_eval(
        fe_cfg, run_id=ctx.run_id, out_dir=(ctx.run_dir / "factor_eval").resolve(),
        compiled_expression_path=arts.compiled_expression_path,
    )
    arts.factor_eval_meta = _relpath(Path(out["factor_eval_meta"]))
    arts.factor_scores_json = _relpath(Path(out["factor_scores_json"]))
    arts.factor_pareto_csv = _relpath(Path(out["pareto_csv"]))


def _step_ml(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    flow_steps.run_ml_training(cfg)


def _step_rl(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    flow_steps.run_rl_training(cfg)


def _step_backtest(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    flow_steps.run_backtest(cfg, ctx.feedback_path)


def _step_tca(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    out = flow_steps.run_tca(cfg, run_id=ctx.run_id, out_dir=(ctx.run_dir / "tca").resolve())
    arts.tca_report = _relpath(Path(out["tca_report"]))
    if out.get("tca_html"):
        arts.tca_html = _relpath(Path(out["tca_html"]))


def _step_report(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    artifact_paths = {
        "capture_manifest": arts.capture_manifest,
        "capture_match_path": arts.capture_match_path,
        "capture_level2_path": arts.capture_level2_path,
        "lob_state_parquet": arts.lob_state_parquet,
        "rebuild_report": arts.rebuild_report,
        "micro_feature_parquet": arts.micro_feature_parquet,
        "micro_feature_manifest": arts.micro_feature_manifest,
        "factor_spec_json": arts.factor_spec_json,
        "factor_ast_json": arts.factor_ast_json,
        "factor_expression_txt": arts.factor_expression_txt,
        "factor_scores_json": arts.factor_scores_json,
        "factor_pareto_csv": arts.factor_pareto_csv,
        "tca_report": arts.tca_report,
        "tca_html": arts.tca_html,
        "feedback_summary": _relpath(ctx.feedback_path),
        "config_path": _relpath(ctx.config_path) if ctx.config_path else None,
    }
    out = flow_steps.run_report_bundle(
        cfg, run_id=ctx.run_id, out_dir=(ctx.run_dir / "bundle").resolve(), artifacts=artifact_paths,
    )
    arts.bundle_zip = _relpath(Path(out["bundle_zip"]))
    arts.bundle_manifest = _relpath(Path(out["bundle_manifest"]))


def _step_strategy_miner(cfg: Dict[str, Any], arts: RunArtifacts, ctx: StepContext) -> None:
    flow_steps.run_strategy_miner_step(cfg, run_id=ctx.run_id, out_dir=(ctx.run_dir / "strategy_miner").resolve())


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
STEP_HANDLERS: Dict[str, Any] = {
    "feature": _step_feature,
    "capture": _step_capture,
    "lob_rebuild": _step_lob_rebuild,
    "micro_feature": _step_micro_feature,
    "portfolio": _step_portfolio,
    "expression": _step_expression,
    "factor_compile": _step_factor_compile,
    "factor_eval": _step_factor_eval,
    "ml": _step_ml,
    "rl": _step_rl,
    "backtest": _step_backtest,
    "tca": _step_tca,
    "report": _step_report,
    "strategy_miner": _step_strategy_miner,
}

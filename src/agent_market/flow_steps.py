from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market.backtest_results import write_latest_backtest_summary
from agent_market import paths

logger = logging.getLogger(__name__)

TrainingPipeline = None
REPO_ROOT = paths.REPO_ROOT


def run_command(cmd: list[str], cwd: Optional[str] = None) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603,S607


def get_freqtrade_version() -> dict[str, Any]:
    wrapper = REPO_ROOT / "scripts" / "freqtrade_cli.py"
    if wrapper.exists():
        cmd = [sys.executable, str(wrapper), "--version"]
        cwd = str(REPO_ROOT)
    else:
        cmd, cwd = _ensure_freqtrade_command(["freqtrade", "--version"], cwd=str(REPO_ROOT))
    try:
        proc = subprocess.run(  # noqa: S603,S607
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "cmd": cmd, "cwd": cwd, "error": str(exc), "version": None}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    version = None
    for line in (stdout.splitlines() + stderr.splitlines()):
        candidate = line.strip()
        if candidate:
            version = candidate
            break

    return {
        "ok": proc.returncode == 0,
        "cmd": cmd,
        "cwd": cwd,
        "returncode": int(proc.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "version": version,
    }


def _resolve_path(path: str | Path, cwd: Optional[str] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    if cwd:
        return (REPO_ROOT / cwd / p).resolve()
    return paths.resolve_repo_path(p)


def _resolve_cwd(cwd: Optional[str]) -> str:
    if not cwd:
        return str(REPO_ROOT)
    return str((REPO_ROOT / cwd).resolve())


def _extract_config_path_from_command(cmd: list[str]) -> Optional[str]:
    for flag in ("--config", "-c"):
        if flag in cmd:
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                return cmd[idx + 1]
    return None


def _resolve_executable(name_or_path: str) -> Optional[str]:
    """Resolve an executable path for subprocess.

    - If an explicit path is provided, verify it exists.
    - If a bare command is provided, search PATH.
    - If still not found, try the venv adjacent binary (sys.executable sibling).
    """
    if not name_or_path:
        return None

    p = Path(name_or_path)
    if p.is_absolute() or any(sep in name_or_path for sep in ("/", "\\")):
        return str(p) if p.exists() else None

    found = shutil.which(name_or_path)
    if found:
        return found

    # If we're running inside a virtualenv, the binary is often next to sys.executable.
    try:
        sibling = Path(sys.executable).with_name(name_or_path)
        if sibling.exists():
            return str(sibling)
    except (ValueError, OSError):
        logger.debug("Failed to check venv sibling for %s", name_or_path)

    # Repo-local venv convention (best-effort).
    for cand in [
        REPO_ROOT / ".venv" / "bin" / name_or_path,
        REPO_ROOT / ".venv" / "Scripts" / (name_or_path + ".exe"),
        REPO_ROOT / "venv" / "bin" / name_or_path,
        REPO_ROOT / "venv" / "Scripts" / (name_or_path + ".exe"),
    ]:
        if cand.exists():
            return str(cand)
    return None


def _inject_userdir(cmd: list[str], userdir: str) -> list[str]:
    if not userdir or "--userdir" in cmd:
        return cmd
    try:
        idx = cmd.index("backtesting") + 1
    except ValueError:
        return cmd
    return cmd[:idx] + ["--userdir", str(userdir)] + cmd[idx:]


def _ensure_freqtrade_command(cmd: list[str], *, cwd: str) -> tuple[list[str], str]:
    """Best-effort make sure cmd[0] is runnable.

    Supports:
    - resolving `freqtrade` to PATH/venv sibling/./.venv/bin/freqtrade
    - falling back to `python -m freqtrade` when the binary cannot be found

    Note: `python -m freqtrade` from repo root can be shadowed by the local `freqtrade/`
    directory, so this fallback uses `cwd=user_data/` when possible.
    """
    if not cmd:
        raise ValueError("Empty command")

    exe = _resolve_executable(str(cmd[0]))
    if exe:
        cmd[0] = exe
        return cmd, cwd

    # Fallback to python -m freqtrade, but avoid CWD shadowing.
    safe_cwd = str(paths.user_data_root())
    cmd = [sys.executable, "-m", "freqtrade"] + cmd[1:]
    return cmd, (safe_cwd if Path(safe_cwd).exists() else cwd)


def run_feature_generation(cfg: Dict[str, Any]) -> None:
    script = Path(cfg.get("script", "scripts/freqai_feature_agent.py"))
    args = list(map(str, cfg.get("args", [])))
    script_path = _resolve_path(script, cwd=cfg.get("cwd"))
    cmd = [sys.executable, str(script_path)] + args
    run_command(cmd, cwd=_resolve_cwd(cfg.get("cwd")))


def run_expression_generation(cfg: Dict[str, Any], feedback_path: Path) -> None:
    script = Path(cfg.get("script", "scripts/freqai_expression_agent.py"))
    args = list(map(str, cfg.get("args", [])))
    fb_path = paths.resolve_repo_path(cfg.get("feedback_path", feedback_path))
    append_feedback = fb_path.exists() and "--feedback" not in args
    if append_feedback:
        args += ["--feedback", str(fb_path)]
        if "--feedback-top" not in args:
            args += ["--feedback-top", str(cfg.get("feedback_top", 10))]
        logger.info("Injecting feedback summary for expression generation: %s", fb_path)
    script_path = _resolve_path(script, cwd=cfg.get("cwd"))
    cmd = [sys.executable, str(script_path)] + args
    run_command(cmd, cwd=_resolve_cwd(cfg.get("cwd")))


def run_ml_training(cfg: Dict[str, Any]) -> None:
    pipeline_cls = TrainingPipeline
    if pipeline_cls is None:
        from agent_market.freqai.training.pipeline import TrainingPipeline as pipeline_cls  # noqa: WPS433

    configs = cfg.get("configs")
    single = cfg.get("config")
    if configs and single:
        raise ValueError("ml_training.config and ml_training.configs are mutually exclusive")
    if configs:
        if not isinstance(configs, list):
            raise ValueError("ml_training.configs must be a list")
        job_list = [item for item in configs if isinstance(item, dict)]
        if len(job_list) != len(configs):
            raise ValueError("ml_training.configs must contain JSON objects only")
    else:
        if not isinstance(single, dict):
            raise ValueError("ml_training.config must be provided as a JSON object")
        job_list = [single]

    total = len(job_list)
    for idx, job_cfg in enumerate(job_list, start=1):
        model_name = job_cfg.get("model", {}).get("name", "unknown")
        logger.info("Starting ML training job %s/%s with model=%s", idx, total, model_name)
        pipeline_cls(job_cfg).run()


def run_rl_training(cfg: Dict[str, Any]) -> None:
    config = cfg.get("config")
    if not isinstance(config, dict):
        raise ValueError("rl_training.config must be provided as a JSON object")
    logger.info("Starting RL training")

    model_dir = paths.resolve_repo_path((config.get("output") or {}).get("model_dir") or str(paths.models_root() / "rl_real"))
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "rl_training_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    script = REPO_ROOT / "scripts" / "train_rl.py"
    cmd = [sys.executable, str(script), "--config", str(config_path)]
    run_command(cmd, cwd=str(REPO_ROOT))


def _maybe_generate_rl_signals_for_backtest(cfg: Dict[str, Any]) -> None:
    if cfg.get("generate_rl_signals") is False:
        logger.info("Skipping RL signal generation (generate_rl_signals=false)")
        return

    rl_summary = cfg.get("rl_summary", str(paths.models_root() / "rl_real" / "training_summary.json"))
    rl_summary_path = paths.resolve_repo_path(str(rl_summary))
    if not rl_summary_path.exists():
        logger.info("RL summary not found (%s); skipping RL signal generation", rl_summary_path)
        return

    cmd_list: Optional[list[str]] = None
    backtest_cwd = cfg.get("cwd")
    if "command" in cfg:
        cmd_list = [str(part) for part in cfg["command"]]
        config_arg = _extract_config_path_from_command(cmd_list)
    else:
        config_arg = str(cfg.get("config")) if cfg.get("config") else None
    if not config_arg:
        logger.info("Backtest config path not found; skipping RL signal generation")
        return

    freqtrade_config_path = _resolve_path(config_arg, cwd=backtest_cwd)
    if not freqtrade_config_path.exists():
        logger.info("Freqtrade config not found (%s); skipping RL signal generation", freqtrade_config_path)
        return

    script = REPO_ROOT / "scripts" / "rl_generate_signals.py"
    gen_cmd = [sys.executable, str(script), "--config", str(freqtrade_config_path)]
    logger.info("Generating RL signals for backtest (%s)", freqtrade_config_path)
    run_command(gen_cmd, cwd=str(REPO_ROOT))


def run_backtest(cfg: Dict[str, Any], feedback_path: Path) -> None:
    _maybe_generate_rl_signals_for_backtest(cfg)

    if "command" in cfg:
        cmd = [str(part) for part in cfg["command"]]
    else:
        binary = cfg.get("binary", "freqtrade")
        cmd = [str(binary), "backtesting"]
        userdir = str(paths.resolve_repo_path(str(cfg.get("userdir") or "user_data")))
        cmd = _inject_userdir(cmd, userdir)
        if cfg.get("config"):
            cmd += ["--config", str(_resolve_path(str(cfg["config"]), cwd=cfg.get("cwd")))]
        if cfg.get("strategy"):
            cmd += ["--strategy", str(cfg["strategy"])]
        if cfg.get("strategy_path"):
            cmd += ["--strategy-path", str(_resolve_path(str(cfg["strategy_path"]), cwd=cfg.get("cwd")))]
        if cfg.get("timerange"):
            cmd += ["--timerange", str(cfg["timerange"])]
        cmd += list(map(str, cfg.get("extra_args", [])))

    cwd = _resolve_cwd(cfg.get("cwd"))
    cmd = _inject_userdir(cmd, str(paths.resolve_repo_path(str(cfg.get("userdir") or "user_data"))))
    wrapper = REPO_ROOT / "scripts" / "freqtrade_cli.py"
    if wrapper.exists() and cmd and Path(str(cmd[0])).name.lower().startswith("freqtrade"):
        cmd = [sys.executable, str(wrapper)] + cmd[1:]
    else:
        cmd, cwd = _ensure_freqtrade_command(cmd, cwd=cwd)
    run_command(cmd, cwd=cwd)
    results_dir = paths.resolve_repo_path(cfg.get("results_dir", "user_data/backtest_results"))
    out_path = paths.resolve_repo_path(cfg.get("feedback_path", feedback_path))
    summary = write_latest_backtest_summary(results_dir, out_path)
    if summary is not None:
        logger.info("Backtest summary written to %s", out_path)


def run_micro_feature(cfg: Dict[str, Any], *, run_id: str, out_dir: Path) -> dict[str, str]:
    """Generate OHLCV-based micro features (Phase 1).

    Required cfg keys:
      - config: freqtrade JSON config path (relative to repo root or absolute)
    Optional:
      - timeframe, timerange, pairs, data_dir
    """
    mode = str(cfg.get("mode") or "").strip().lower()
    lob_state = cfg.get("lob_state") or cfg.get("lob_state_parquet")
    match_path = cfg.get("match") or cfg.get("match_path")
    if mode == "microstructure" or lob_state or match_path:
        if not lob_state or not match_path:
            raise ValueError("micro_feature microstructure mode requires lob_state and match")

        symbol = cfg.get("symbol")
        depth_levels = int(cfg.get("depth_levels") or cfg.get("depth") or 20)
        windows_raw = cfg.get("windows_sec") or cfg.get("windows") or [10]
        if isinstance(windows_raw, str):
            windows = [int(x) for x in windows_raw.replace(",", " ").split() if x.strip()]
        elif isinstance(windows_raw, (int, float)):
            windows = [int(windows_raw)]
        elif isinstance(windows_raw, list):
            windows = [int(x) for x in windows_raw if int(x) > 0]
        else:
            windows = [10]
        if not windows:
            windows = [10]

        from agent_market.microstructure.micro_feature import (  # noqa: WPS433
            generate_microstructure_features_from_lob_and_match,
        )

        outputs = generate_microstructure_features_from_lob_and_match(
            root=REPO_ROOT,
            run_id=str(run_id),
            out_dir=Path(out_dir),
            lob_state=_resolve_path(str(lob_state), cwd=cfg.get("cwd")),
            match_path=_resolve_path(str(match_path), cwd=cfg.get("cwd")),
            symbol=str(symbol) if symbol else None,
            depth_levels=int(depth_levels),
            windows_sec=windows,
        )
        return {
            "features_parquet": str(outputs.features_parquet),
            "manifest_json": str(outputs.manifest_json),
        }

    config_path = cfg.get("config") or cfg.get("freqtrade_config")
    if not config_path:
        raise ValueError("micro_feature.config is required (or set mode=microstructure)")

    timeframe = cfg.get("timeframe")
    timerange = cfg.get("timerange")
    pairs = cfg.get("pairs")
    if isinstance(pairs, str):
        pairs = [p for p in pairs.replace(",", " ").split() if p.strip()]
    if pairs is not None and not isinstance(pairs, list):
        raise ValueError("micro_feature.pairs must be a list or string")

    from agent_market.microstructure.micro_feature import (  # noqa: WPS433
        generate_micro_features_from_freqtrade_config,
    )

    outputs = generate_micro_features_from_freqtrade_config(
        root=REPO_ROOT,
        freqtrade_config=_resolve_path(str(config_path), cwd=cfg.get("cwd")),
        run_id=str(run_id),
        out_dir=Path(out_dir),
        timeframe=str(timeframe) if timeframe else None,
        timerange=str(timerange) if timerange else None,
        pairs=[str(p) for p in pairs] if pairs else None,
        data_dir=cfg.get("data_dir"),
    )
    return {
        "features_parquet": str(outputs.features_parquet),
        "manifest_json": str(outputs.manifest_json),
    }


def run_tca(cfg: Dict[str, Any], *, run_id: str, out_dir: Path) -> dict[str, str]:
    """Generate TCA report from latest backtest zip (Phase 1)."""
    from agent_market.backtest_results import find_latest_backtest_zip  # noqa: WPS433
    from agent_market.tca.report import generate_tca_report  # noqa: WPS433

    results_dir = cfg.get("results_dir") or str(paths.user_data_root() / "backtest_results")
    zp_override = cfg.get("backtest_zip") or cfg.get("zip")
    if zp_override:
        zip_path = _resolve_path(str(zp_override), cwd=cfg.get("cwd"))
    else:
        zip_path = find_latest_backtest_zip(_resolve_path(str(results_dir), cwd=cfg.get("cwd")))
        if zip_path is None:
            raise FileNotFoundError(f"No backtest-result-*.zip found in {results_dir}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / "tca_report.json").resolve()

    html = bool(cfg.get("html"))
    html_path = (out_dir / "tca_report.html").resolve() if html else None
    generate_tca_report(zip_path, run_id=str(run_id), out_path=out_path, root=REPO_ROOT, html_path=html_path)
    return {
        "tca_report": str(out_path),
        "tca_html": str(html_path) if html_path is not None else "",
        "backtest_zip": str(zip_path),
    }


def run_factor_compile(cfg: Dict[str, Any], *, run_id: str, out_dir: Path) -> dict[str, str]:
    """Compile a FactorSpec into an ExpressionEngine expression (Phase 1)."""
    spec_obj = cfg.get("spec")
    spec_path_cfg = cfg.get("spec_path") or cfg.get("spec_file")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(spec_obj, dict):
        spec_path = (out_dir / "input_factor_spec.json").resolve()
        spec_path.write_text(json.dumps(spec_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    elif spec_path_cfg:
        spec_path = _resolve_path(str(spec_path_cfg), cwd=cfg.get("cwd"))
        if not spec_path.exists():
            raise FileNotFoundError(f"FactorSpec not found: {spec_path}")
    else:
        raise ValueError("factor_compile.spec (object) or factor_compile.spec_path must be provided")

    script = (REPO_ROOT / "scripts" / "factor_compile.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"factor_compile script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--spec",
        str(spec_path),
        "--out-dir",
        str(out_dir.resolve()),
    ]
    run_command(cmd, cwd=str(REPO_ROOT))
    return {
        "factor_spec_json": str((out_dir / "factor_spec.json").resolve()),
        "factor_ast_json": str((out_dir / "factor_ast.json").resolve()),
        "compiled_expression_txt": str((out_dir / "compiled_expression.txt").resolve()),
        "compiled_expression_json": str((out_dir / "compiled_expression.json").resolve()),
    }


def run_factor_eval(
    cfg: Dict[str, Any],
    *,
    run_id: str,
    out_dir: Path,
    compiled_expression_path: Optional[Path] = None,
) -> dict[str, str]:
    """Evaluate a compiled factor expression and write score artifacts (minimal offline mode)."""
    expr_text = cfg.get("expression")
    expr_path_cfg = cfg.get("expression_path") or cfg.get("expression_file")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_path: Optional[Path] = None
    if isinstance(expr_text, str) and expr_text.strip():
        expr_path = (out_dir / "input_expression.txt").resolve()
        expr_path.write_text(expr_text.strip(), encoding="utf-8")
    elif expr_path_cfg:
        expr_path = _resolve_path(str(expr_path_cfg), cwd=cfg.get("cwd"))
        if not expr_path.exists():
            raise FileNotFoundError(f"Expression file not found: {expr_path}")
    elif compiled_expression_path is not None and Path(compiled_expression_path).exists():
        expr_path = Path(compiled_expression_path).resolve()
    else:
        raise ValueError(
            "factor_eval.expression (string) or factor_eval.expression_path must be provided "
            "(or run after factor_compile)"
        )

    script = (REPO_ROOT / "scripts" / "factor_eval.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"factor_eval script not found: {script}")

    rows = int(cfg.get("rows") or 256)
    seed = int(cfg.get("seed") or 7)
    cmd = [
        sys.executable,
        str(script),
        "--expression-path",
        str(expr_path),
        "--out-dir",
        str(out_dir.resolve()),
        "--rows",
        str(rows),
        "--seed",
        str(seed),
    ]
    features_parquet = cfg.get("features_parquet") or cfg.get("features_path") or cfg.get("micro_features_parquet")
    if features_parquet:
        cmd += ["--features-parquet", str(_resolve_path(str(features_parquet), cwd=cfg.get("cwd")))]
    run_command(cmd, cwd=str(REPO_ROOT))

    return {
        "factor_eval_meta": str((out_dir / "factor_eval_meta.json").resolve()),
        "factor_scores_json": str((out_dir / "factor_scores.json").resolve()),
        "pareto_csv": str((out_dir / "pareto.csv").resolve()),
    }


def run_capture(cfg: Dict[str, Any], *, out_dir: Path) -> dict[str, str]:
    """Capture microstructure data (fixture or live)."""
    exchange = str(cfg.get("exchange") or "kucoin").strip().lower()
    channels = str(cfg.get("channels") or "match,level2").strip()
    symbols = str(cfg.get("symbols") or "BTC-USDT").strip()
    duration_sec = int(cfg.get("duration_sec") or 60)
    fixture = cfg.get("fixture")

    script = (REPO_ROOT / "scripts" / "micro_capture.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"micro_capture script not found: {script}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--exchange",
        exchange,
        "--channels",
        channels,
        "--out-dir",
        str(out_dir.resolve()),
    ]
    if fixture:
        cmd += ["--fixture", str(_resolve_path(str(fixture), cwd=cfg.get("cwd")))]
    else:
        cmd += ["--symbols", symbols, "--duration-sec", str(duration_sec)]

    run_command(cmd, cwd=str(REPO_ROOT))
    return {
        "capture_dir": str(out_dir.resolve()),
        "manifest_json": str((out_dir / "manifest.json").resolve()),
        "match_path": str((out_dir / "match.ndjson.gz").resolve()),
        "level2_path": str((out_dir / "level2.ndjson.gz").resolve()),
    }


def run_lob_rebuild(cfg: Dict[str, Any], *, capture_dir: Path, out_dir: Path) -> dict[str, str]:
    """Rebuild LOB state from capture dir + snapshot fixture."""
    exchange = str(cfg.get("exchange") or "kucoin").strip().lower()
    symbol = str(cfg.get("symbol") or "BTC-USDT").strip()
    depth = int(cfg.get("depth") or 20)
    snapshot = cfg.get("snapshot")
    if not snapshot:
        raise ValueError("lob_rebuild.snapshot must be provided")

    script = (REPO_ROOT / "scripts" / "lob_rebuild.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"lob_rebuild script not found: {script}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = _resolve_path(str(snapshot), cwd=cfg.get("cwd"))
    if not snapshot_path.exists():
        raise FileNotFoundError(f"LOB snapshot not found: {snapshot_path}")

    cmd = [
        sys.executable,
        str(script),
        "--capture-dir",
        str(Path(capture_dir).resolve()),
        "--snapshot",
        str(snapshot_path),
        "--symbol",
        symbol,
        "--depth",
        str(depth),
        "--out-dir",
        str(out_dir.resolve()),
    ]
    run_command(cmd, cwd=str(REPO_ROOT))
    return {
        "exchange": exchange,
        "symbol": symbol,
        "lob_state_parquet": str((out_dir / "lob_state.parquet").resolve()),
        "rebuild_report_json": str((out_dir / "rebuild_report.json").resolve()),
    }


def run_report_bundle(
    cfg: Dict[str, Any],
    *,
    run_id: str,
    out_dir: Path,
    artifacts: Dict[str, Optional[str]],
) -> dict[str, str]:
    """Bundle key artifacts into a single zip for sharing/downloading."""
    import zipfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_zip = (out_dir / "bundle.zip").resolve()
    manifest_path = (out_dir / "bundle_manifest.json").resolve()

    included: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for key, raw in (artifacts or {}).items():
        if not raw:
            continue
        try:
            p = _resolve_path(str(raw), cwd=cfg.get("cwd"))
        except (ValueError, OSError):
            logger.debug("Could not resolve artifact path %r, using as-is", raw)
            p = Path(str(raw))
        if not p.exists():
            missing.append({"key": str(key), "path": str(raw)})
            continue
        try:
            arc = str(p.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            arc = p.name
        included.append({"key": str(key), "path": str(p), "arcname": arc})

    with zipfile.ZipFile(bundle_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in included:
            zf.write(item["path"], arcname=item["arcname"])
        zf.writestr(
            "bundle_manifest.json",
            json.dumps(
                {
                    "version": 1,
                    "run_id": str(run_id),
                    "included": included,
                    "missing": missing,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_id": str(run_id),
                "bundle_zip": str(bundle_zip),
                "included": included,
                "missing": missing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"bundle_zip": str(bundle_zip), "bundle_manifest": str(manifest_path)}


def run_strategy_miner_step(cfg: Dict[str, Any], *, run_id: str, out_dir: Path) -> dict[str, str]:
    """Run strategy miner as a flow step."""
    from agent_market.strategy_miner.dtypes import MinerConfig  # noqa: WPS433
    from agent_market.strategy_miner.runner import run_strategy_miner  # noqa: WPS433

    miner_cfg = MinerConfig.from_dict(cfg)

    resume_path = cfg.get("resume")
    resume = Path(resume_path) if resume_path else None

    state = run_strategy_miner(miner_cfg, run_id=run_id, resume=resume)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "strategy_miner_summary.json"
    summary_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "strategy_miner_summary": str(summary_path),
        "strategy_miner_dir": str(out_dir.resolve()),
        "run_id": state.run_id,
        "best_score": str(state.best_score),
        "best_strategy": state.best_candidate.name if state.best_candidate else "",
    }


__all__ = [
    "run_backtest",
    "run_command",
    "run_expression_generation",
    "run_feature_generation",
    "get_freqtrade_version",
    "run_ml_training",
    "run_rl_training",
    "run_micro_feature",
    "run_tca",
    "run_factor_compile",
    "run_factor_eval",
    "run_capture",
    "run_lob_rebuild",
    "run_report_bundle",
    "run_strategy_miner_step",
]

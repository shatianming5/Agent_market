#!/usr/bin/env python3
"""Factor Lab — unified CLI for factor mining / validation / backtest / deployment.

Subcommands:
  data      download raw OHLCV / funding data
  features  merge engineered features (mtf4h / xs / pair / funding / micro)
  mine      run iterative factor mining (IC + composition + optional LLM)
  validate  sub-period stability + random baseline audit
  backtest  walk-forward (training + freqtrade backtesting)
  deploy    list / switch / describe factor libraries

Examples:
  # Full data bootstrap (one-time, ~2-4 hours)
  python scripts/factor_lab.py data kucoin --timeframe 1h --years 3
  python scripts/factor_lab.py data kucoin --timeframe 4h --years 3
  python scripts/factor_lab.py data funding
  python scripts/factor_lab.py data okx-futures

  # Merge all feature types into 1h feathers
  python scripts/factor_lab.py features all

  # Run 50-round mining (no LLM, Python-only — fast)
  python scripts/factor_lab.py mine --tag exp1 --rounds 50

  # Run 200-round mining with GPT-5.4 (via cli-proxy)
  python scripts/factor_lab.py mine --tag exp2 --rounds 200 --llm

  # Export top-30 from a mining run
  python scripts/factor_lab.py mine-export --tag exp1 --n 30

  # Validate a factor library
  python scripts/factor_lab.py validate user_data/freqai_expressions_exp1.json

  # Walk-forward with a specific factor library
  python scripts/factor_lab.py deploy switch freqai_expressions_exp1.json
  python scripts/factor_lab.py backtest --tag exp1 --train-months 6

  # Deploy management
  python scripts/factor_lab.py deploy list
  python scripts/factor_lab.py deploy describe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src is in path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# Load .env
env_file = ROOT / ".env"
if env_file.exists():
    import os
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from agent_market import paths as repo_paths
from agent_market.factor_lab import data, features, mining, validation, backtest, deploy, combo_ga, rl, reporting, rank_portfolio, strategy_loop, okx_universe, lean_bridge, mine_lean_gate
from agent_market.factor_lab.cache import DEFAULT_CACHE_DIR, cache_inventory, clear_cache
from agent_market.factor_lab.timeframes import bps_to_rate, normalize_lane, parse_label_horizons, primary_label_horizon
from agent_market.factor_memory import audit_factor_memory_path

FUTURES_VENUE_CHOICES = ["okx", "bybit", "binance"]
DATA_VENUE_CHOICES = ["auto", "kucoin", "okx", "bybit", "binance"]
MINING_DATA_VENUE_CHOICES = ["kucoin", "okx", "bybit", "binance"]


# ============================================================
# Subcommand handlers
# ============================================================

def cmd_data(args):
    if args.source == "kucoin":
        end = args.end or "2026-04-18"
        start = args.start
        if not start and args.years:
            from datetime import datetime, timedelta
            start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")
        start = start or "2023-04-12"
        data.download_kucoin(timeframe=args.timeframe, start=start, end=end, pairs=args.pairs, sleep_s=args.sleep)
    elif args.source == "okx-futures":
        pairs = args.pairs
        pair_start_dates = None
        if args.universe:
            universe_pairs, universe_starts = okx_universe.manifest_pairs_and_starts(args.universe)
            if pairs:
                selected = [part.strip() for part in str(pairs).replace(";", ",").split(",") if part.strip()]
                pairs = selected
                pair_start_dates = {pair: universe_starts[pair] for pair in selected if pair in universe_starts}
            else:
                pairs = universe_pairs
                pair_start_dates = universe_starts
        if args.aux_only:
            print(json.dumps(data.prepare_okx_futures_auxiliary(data_dir=Path(args.data_dir) if args.data_dir else data.OKX_FUTURES_DIR, timeframe=args.timeframe), indent=2))
        else:
            data.download_okx_futures(
                start=args.start or "2025-04-12",
                end=args.end or "2026-04-12",
                timeframe=args.timeframe,
                pairs=pairs,
                sleep_s=args.sleep,
                data_dir=Path(args.data_dir) if args.data_dir else data.OKX_FUTURES_DIR,
                pair_start_dates=pair_start_dates,
            )
    elif args.source == "bybit-futures":
        if args.aux_only:
            raise SystemExit("--aux-only only applies to okx-futures")
        data.download_bybit_futures(
            start=args.start or "2025-04-12",
            end=args.end or "2026-04-12",
            timeframe=args.timeframe,
            pairs=args.pairs,
            sleep_s=args.sleep,
            data_dir=Path(args.data_dir) if args.data_dir else data.BYBIT_FUTURES_DIR,
        )
    elif args.source == "binance-futures":
        if args.aux_only:
            raise SystemExit("--aux-only only applies to okx-futures")
        data.download_binance_futures(
            start=args.start or "2025-04-12",
            end=args.end or "2026-04-12",
            timeframe=args.timeframe,
            pairs=args.pairs,
            sleep_s=args.sleep,
            data_dir=Path(args.data_dir) if args.data_dir else data.BINANCE_FUTURES_DIR,
        )
    elif args.source == "okx-universe":
        if args.universe:
            path = okx_universe.write_okx_universe_manifest(
                args.universe,
                full_history_start=args.full_history_start,
                top_n=args.top_n,
            )
            payload = okx_universe.load_okx_universe_manifest(path)
            print(json.dumps({
                "path": str(path),
                "name": payload.get("name"),
                "pair_count": payload.get("pair_count"),
                "spot_matched_count": payload.get("spot_matched_count"),
                "min_list_date": payload.get("min_list_date"),
                "max_list_date": payload.get("max_list_date"),
            }, indent=2))
        else:
            paths = okx_universe.build_default_okx_universe_manifests(full_history_start=args.full_history_start)
            print(json.dumps([str(path) for path in paths], indent=2))
    elif args.source == "funding":
        data.download_funding(start=args.start or "2023-04-12", end=args.end or "2026-04-18")


def cmd_features(args):
    kinds = args.kinds
    if "all" in kinds: kinds = ["mtf4h", "xs", "pair", "funding", "micro", "ohlcv_micro"]
    feature_kwargs = {"pairs": args.pairs, "data_dir": args.data_dir}
    if "mtf4h" in kinds:
        print("=== merge mtf4h ==="); r = features.merge_mtf4h(**feature_kwargs)
        for k, v in r.items(): print(f"  {k}: {v}")
    if "xs" in kinds:
        print("\n=== merge cross-sectional ==="); r = features.merge_cross_sectional(**feature_kwargs)
        for k, v in r.items(): print(f"  {k}: {v}")
    if "pair" in kinds:
        print("\n=== merge pair-relative ===")
        r = features.merge_pair_relative(
            reference_pairs=args.pair_reference,
            beta_window=args.pair_beta_window,
            **feature_kwargs,
        )
        for k, v in r.items(): print(f"  {k}: {v}")
    if "funding" in kinds:
        print("\n=== merge funding ==="); r = features.merge_funding(**feature_kwargs)
        for k, v in r.items(): print(f"  {k}: {v}")
    if "micro" in kinds:
        print("\n=== merge micro ==="); r = features.merge_micro(**feature_kwargs)
        for k, v in r.items(): print(f"  {k}: {v}")
    if "ohlcv_micro" in kinds:
        print("\n=== merge ohlcv_micro ==="); r = features.merge_ohlcv_micro(**feature_kwargs)
        for k, v in r.items(): print(f"  {k}: {v}")
    if "microstructure" in kinds:
        if not args.microstructure_parquet or not args.microstructure_target:
            raise SystemExit("--microstructure-parquet and --microstructure-target are required for kind=microstructure")
        print("\n=== merge microstructure parquet → 1h feather ===")
        out = features.merge_microstructure_parquet(
            features_parquet=Path(args.microstructure_parquet),
            target_feather=Path(args.microstructure_target),
            symbol=str(args.microstructure_symbol) if args.microstructure_symbol else None,
            agg=str(args.microstructure_agg),
            prefix=str(args.microstructure_prefix or ""),
        )
        print(out)


def cmd_features_restore(args):
    features.restore_backups(kind=args.kind, pairs=args.pairs, data_dir=args.data_dir)


def cmd_mine(args):
    lane_tf = args.timeframe
    if str(args.lane or "auto") not in {"auto", "1h"} and str(args.timeframe or "1h") == "1h":
        lane_tf = None
    lane = normalize_lane(args.lane, timeframe=lane_tf)
    label_horizons = parse_label_horizons(args.label_horizons, default=lane.label_horizons)
    fee_rate = bps_to_rate(args.fee_bps) if args.fee_bps is not None else float(args.fee_rate)
    slippage = bps_to_rate(args.slippage_bps) if args.slippage_bps is not None else float(args.slippage)
    label_period = primary_label_horizon(label_horizons, default=lane.label_horizons)
    cfg = mining.MiningConfig(
        rounds=args.rounds, top_k=args.top_k,
        llm_per_loop=args.llm_per_loop, py_per_loop=args.py_per_loop,
        ic_gate=args.ic_gate, sign_gate=args.sign_gate,
        novelty_gate=args.novelty_gate,
        hard_corr_gate=args.hard_corr_gate,
        soft_corr_penalty_start=args.soft_corr_penalty_start,
        max_same_family_in_top40=args.max_same_family_in_top40,
        max_same_signature=args.max_same_signature,
        checkpoint_every=args.checkpoint_every,
        use_llm=args.llm, llm_required=args.llm_required,
        llm_timeout=args.llm_timeout, llm_retries=args.llm_retries,
        llm_max_tokens=args.llm_max_tokens,
        llm_reasoning_effort=args.llm_reasoning_effort,
        timeframe=lane.timeframe,
        evaluation_lane=lane.lane,
        data_venue=args.data_venue,
        label_horizons=label_horizons,
        label_period=label_period,
        embargo_bars=args.embargo_bars or lane.embargo_bars,
        micro_data_quality=args.micro_data_quality,
        eval_mode=args.eval_mode,
        xs_weight=args.xs_weight,
        turnover_weight=args.turnover_weight,
        stability_mode=args.stability_mode,
        fee_rate=fee_rate,
        slippage=slippage,
        label_mode=args.label_mode,
        pair_reference=args.pair_reference,
        data_dir=args.data_dir,
        pairs=args.pairs,
        seed_file=args.seed_file,
        purify_mode=args.purify_mode,
        purify_winsor=args.purify_winsor,
        purify_standardize=args.purify_standardize,
        purify_neutralize=args.purify_neutralize,
        purify_exposures=args.purify_exposures,
        alpha_objective=args.alpha_objective,
        prompt_profile=args.prompt_profile,
        llm_filter_low_coverage=not args.llm_no_feature_filter,
        llm_min_feature_coverage=args.llm_min_feature_coverage,
        llm_min_feature_rows=args.llm_min_feature_rows,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        lean_gate_every=args.lean_gate_every,
        lean_gate_fail_fast=args.lean_gate_fail_fast,
        lean_gate_force=not args.lean_gate_no_force,
        lean_gate_n=args.lean_gate_n,
        lean_gate_venue=args.lean_gate_venue,
        lean_gate_data_venue=args.lean_gate_data_venue,
        lean_gate_start=args.lean_gate_start,
        lean_gate_end=args.lean_gate_end,
        lean_gate_bin=args.lean_gate_bin,
        lean_gate_timeout=args.lean_gate_timeout,
        lean_gate_data_root=args.lean_gate_data_root,
        lean_gate_required_status=args.lean_gate_required_status,
        lean_gate_min_final_equity=args.lean_gate_min_final_equity,
        lean_gate_max_drawdown_pct=args.lean_gate_max_drawdown_pct,
        lean_gate_min_trades=args.lean_gate_min_trades,
        lean_gate_rank_top_k=args.lean_gate_rank_top_k,
        lean_gate_gross_cap=args.lean_gate_gross_cap,
        lean_gate_net_cap=args.lean_gate_net_cap,
        lean_gate_single_pair_cap=args.lean_gate_single_pair_cap,
        lean_gate_side_mode=args.lean_gate_side_mode,
        lean_gate_score_threshold=args.lean_gate_score_threshold,
        lean_gate_rebalance_hours=args.lean_gate_rebalance_hours,
        lean_gate_rebalance_minutes=args.lean_gate_rebalance_minutes,
        lean_gate_risk_per_trade=args.lean_gate_risk_per_trade,
        lean_gate_leverage_cap=args.lean_gate_leverage_cap,
        lean_gate_recompute_corr=args.lean_gate_recompute_corr,
    )
    survivors = mining.mine(cfg, tag=args.tag, resume=args.resume)
    print(f"\n[mining] final: {len(survivors)} survivors")
    for i, s in enumerate(sorted(survivors, key=lambda x: mining._portfolio_key(x, "portfolio"), reverse=True)[:10]):  # noqa: SLF001
        if args.alpha_objective == "pure_residual":
            print(
                f"  #{i+1} neutral_ic={s.neutralized_ic:+.4f} residual={s.residual_ic_ratio:.2f} "
                f"r2={s.exposure_r2:.2f} sig={s.sign_agree} [{s.origin}]  {s.expression[:80]}"
            )
        else:
            print(f"  #{i+1} ic={s.oos_ic:+.3f} sig={s.sign_agree}/10 [{s.origin}]  {s.expression[:80]}")


def cmd_mine_export(args):
    out = mining.export_top(
        tag=args.tag,
        n=args.n,
        diverse=args.diverse,
        corr_gate=args.corr_gate,
        score_mode=args.score_mode,
        family_max=args.family_max,
        timeframe=args.timeframe,
        evaluation_lane=args.lane,
        data_venue=args.data_venue,
        label_horizons=args.label_horizons,
        eval_mode=args.eval_mode,
        label_mode=args.label_mode,
        pair_reference=args.pair_reference,
        data_dir=args.data_dir,
        pairs=args.pairs,
        purify_mode=args.purify_mode,
        purify_winsor=args.purify_winsor,
        purify_standardize=args.purify_standardize,
        purify_neutralize=args.purify_neutralize,
        purify_exposures=args.purify_exposures,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )
    print(f"Wrote {out}")
    if args.diverse:
        from agent_market.factor_lab.paths import USER_DATA
        print(f"Wrote {USER_DATA / f'factor_diversity_report_{args.tag}.json'}")


def _rank_gate_kwargs_from_args(args):
    return {
        "top_k": args.top_k,
        "gross_cap": args.gross_cap,
        "net_cap": args.net_cap,
        "single_pair_cap": args.single_pair_cap,
        "risk_profile": args.risk_profile,
        "side_mode": args.side_mode,
        "min_abs_score_z": args.score_threshold,
        "rebalance_hours": args.rebalance_hours,
        "rebalance_minutes": args.rebalance_minutes,
        "risk_per_trade": args.risk_per_trade,
        "leverage_cap": args.leverage_cap,
        "edge_mode": args.edge_mode,
        "edge_lookback_hours": args.edge_lookback_hours,
        "edge_min_periods": args.edge_min_periods,
        "edge_deadband": args.edge_deadband,
        "pair_edge_leverage": args.pair_edge_leverage,
        "pair_edge_deadband": args.pair_edge_deadband,
        "pair_edge_strong_ic": args.pair_edge_strong_ic,
        "pair_edge_very_strong_ic": args.pair_edge_very_strong_ic,
        "pair_edge_weak_cap": args.pair_edge_weak_cap,
        "regime_mode": args.regime_mode,
        "regime_min_edge_ic": args.regime_min_edge_ic,
        "regime_min_pair_edge_ic": args.regime_min_pair_edge_ic,
        "regime_min_pair_count": args.regime_min_pair_count,
        "regime_short_max_market_mom_24h": args.regime_short_max_market_mom_24h,
        "regime_short_max_market_mom_72h": args.regime_short_max_market_mom_72h,
        "regime_max_market_atr_pct": args.regime_max_market_atr_pct,
        "short_max_mom_24h": args.short_max_mom_24h,
        "short_max_mom_72h": args.short_max_mom_72h,
        "long_min_mom_24h": args.long_min_mom_24h,
        "max_entry_atr_pct": args.max_entry_atr_pct,
        "short_max_market_mom_24h": args.short_max_market_mom_24h,
        "short_max_market_mom_72h": args.short_max_market_mom_72h,
        "short_max_market_ma_gap": args.short_max_market_ma_gap,
        "short_exit_mom_24h": args.short_exit_mom_24h,
        "short_exit_mom_72h": args.short_exit_mom_72h,
        "short_exit_market_mom_24h": args.short_exit_market_mom_24h,
        "short_exit_market_ma_gap": args.short_exit_market_ma_gap,
        "exclude_pairs": args.exclude_pairs,
        "recompute_corr": not args.no_corr_recompute,
    }


def cmd_mine_lean_gate(args):
    result = mine_lean_gate.run_mine_lean_gate(
        tag=args.tag,
        n=args.n,
        candidate_state=args.candidate_state,
        run_id=args.run_id,
        output=args.output,
        rank_tag=args.rank_tag,
        venue=args.venue,
        timeframe=args.timeframe,
        data_venue=args.data_venue,
        pairs=args.pairs,
        start=args.start,
        end=args.end,
        lean_bin=args.lean_bin,
        lean_timeout=args.lean_timeout,
        lean_data_root=args.lean_data_root,
        lean_required_status=args.lean_required_status,
        min_final_equity=args.min_final_equity,
        max_drawdown_pct=args.max_drawdown_pct,
        min_trades=args.min_trades,
        force=args.force,
        rank_kwargs=_rank_gate_kwargs_from_args(args),
    )
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") != mine_lean_gate.STATUS_PASSED and not args.no_fail_on_reject:
        raise SystemExit(2)


def cmd_validate(args):
    r = validation.validate(Path(args.factors))
    print(f"\nValidation JSON: {json.dumps(r, indent=2)}")


def cmd_factor_report(args):
    result = reporting.factor_report(
        tag=args.tag,
        n=args.n,
        purify_mode=args.purify_mode,
        purify_winsor=args.purify_winsor,
        purify_standardize=args.purify_standardize,
        purify_neutralize=args.purify_neutralize,
        purify_exposures=args.purify_exposures,
        timeframe=args.timeframe,
        label_bars=args.label_bars,
        label_mode=args.label_mode,
        pair_reference=args.pair_reference,
        data_dir=args.data_dir,
        data_venue=args.data_venue,
        pairs=args.pairs,
        score_mode=args.score_mode,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )
    print(json.dumps(result, indent=2))


def cmd_exposure_report(args):
    result = reporting.exposure_report(
        tag=args.tag,
        n=args.n,
        purify_mode=args.purify_mode,
        purify_winsor=args.purify_winsor,
        purify_standardize=args.purify_standardize,
        purify_neutralize=args.purify_neutralize,
        purify_exposures=args.purify_exposures,
        timeframe=args.timeframe,
        label_bars=args.label_bars,
        label_mode=args.label_mode,
        pair_reference=args.pair_reference,
        data_dir=args.data_dir,
        data_venue=args.data_venue,
        pairs=args.pairs,
        score_mode=args.score_mode,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        attribution_mode=args.attribution_mode,
        attribution_max_dates=args.attribution_max_dates,
        attribution_max_exposures=args.attribution_max_exposures,
    )
    print(json.dumps(result, indent=2))


def cmd_cache(args):
    if args.action == "stats":
        print(json.dumps(cache_inventory(args.cache_dir), indent=2))
    elif args.action == "clear":
        print(json.dumps(clear_cache(args.cache_dir), indent=2))


def cmd_memory_audit(args):
    path = repo_paths.resolve_repo_path(args.path) if args.path else repo_paths.global_factor_memory_path()
    result = audit_factor_memory_path(path, write_tags=bool(args.write_tags))
    print(json.dumps(result, indent=2, default=str))


def cmd_backtest(args):
    from datetime import datetime as _dt
    anchor = _dt.strptime(args.anchor, "%Y-%m-%d") if args.anchor else None
    data_start = _dt.strptime(args.data_start, "%Y-%m-%d") if args.data_start else None
    data_end = _dt.strptime(args.data_end, "%Y-%m-%d") if args.data_end else None
    kwargs = {}
    if data_start is not None: kwargs["data_start"] = data_start
    if data_end is not None: kwargs["data_end"] = data_end
    model_params = None
    if args.n_estimators > 0:
        # Use default LGB params but override the tree count knobs
        base = json.loads(json.dumps(backtest.DEFAULT_BASE_CFG))
        model_params = dict(base['model']['params'])
        model_params['n_estimators'] = args.n_estimators
        model_params['num_boost_round'] = args.n_estimators
    results = backtest.run_walkforward(
        tag=args.tag, train_months=args.train_months,
        strategy=args.strategy, ft_config=args.ft_config,
        model_name=args.model,
        model_params=model_params,
        test_start_anchor=anchor,
        num_windows=args.num_windows if args.num_windows > 0 else None,
        expressions_file=args.expressions_file,
        exit_label_period=(args.exit_label_period if args.exit_label_period > 0 else None),
        datadir=args.datadir,
        **kwargs,
    )
    out = ROOT / "artifacts" / f"walkforward_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out}")


def cmd_deploy(args):
    if args.action == "list":
        for lib in deploy.list_factor_libs():
            marker = "⭐" if lib.get("is_current") else "  "
            print(f"{marker} {lib['name']:<50} factors={lib.get('factors','?'):>4}  "
                  f"v={lib.get('version','?'):<15}  {lib.get('meta','')[:40]}")
    elif args.action == "current":
        print(json.dumps(deploy.current_deployment(), indent=2))
    elif args.action == "switch":
        r = deploy.switch_to(args.name)
        print(f"Deployed {r['n_factors']} factors from {r['deployed_from']}")
    elif args.action == "describe":
        print(json.dumps(deploy.describe(args.name), indent=2, default=str))


def cmd_rl(args):
    pairs = args.pairs.split(",") if args.pairs else None
    if args.action == "bc-pretrain":
        from agent_market.factor_lab import rl_bc
        summary = rl_bc.pretrain(
            tag=args.tag, expressions_file=args.expressions,
            timeframe=args.timeframe, epochs=args.bc_epochs,
        )
        print(json.dumps(summary, indent=2, default=str))
        return
    if args.action == "bc-eval":
        from agent_market.factor_lab import rl_bc
        summary = rl_bc.evaluate_bc(
            tag=args.tag, timerange_start=args.timerange_start,
            timerange_end=args.timerange_end,
            expressions_file=args.expressions, timeframe=args.timeframe,
        )
        print(json.dumps(summary, indent=2, default=str))
        return
    if args.action == "train":
        summary = rl.train(
            tag=args.tag,
            expressions_file=args.expressions,
            timeframe=args.timeframe,
            total_timesteps=args.timesteps,
            window_size=args.window_size,
            reward_profile=args.reward_profile,
            env_class=args.env_class,
            algo_class=args.algo_class,
            pairs=pairs,
            policy=args.policy,
        )
    else:
        summary = rl.evaluate(
            tag=args.tag,
            timerange_start=args.timerange_start,
            timerange_end=args.timerange_end,
            expressions_file=args.expressions,
            timeframe=args.timeframe,
            window_size=args.window_size,
            reward_profile=args.reward_profile,
            env_class=args.env_class,
        )
    print()
    print(json.dumps(summary, indent=2, default=str))


def cmd_combo(args):
    summary = combo_ga.run(
        combo_size=args.combo_size, pool_size=args.pool_size,
        min_abs_ic=args.min_abs_ic,
        population=args.population, generations=args.generations,
        novelty_gate=args.jaccard_gate, dedupe_gate=args.dedupe_gate,
        timeframe=args.timeframe,
        seed=args.seed,
        tag=args.tag, export_top_n=args.top_n,
        require_clean=not args.include_snooped,
    )
    print()
    print(json.dumps(summary, indent=2, default=str)[:800])


def cmd_rank_export(args):
    summary = rank_portfolio.rank_export(
        tag=args.tag,
        n=args.n,
        risk_profile=args.risk_profile,
        venue=args.venue,
        timeframe=args.timeframe,
        data_venue=args.data_venue,
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        gross_cap=args.gross_cap,
        net_cap=args.net_cap,
        single_pair_cap=args.single_pair_cap,
        side_mode=args.side_mode,
        min_abs_score_z=args.score_threshold,
        rebalance_hours=args.rebalance_hours,
        rebalance_minutes=args.rebalance_minutes,
        risk_per_trade=args.risk_per_trade,
        leverage_cap=args.leverage_cap,
        edge_mode=args.edge_mode,
        edge_lookback_hours=args.edge_lookback_hours,
        edge_min_periods=args.edge_min_periods,
        edge_deadband=args.edge_deadband,
        pair_edge_leverage=args.pair_edge_leverage,
        pair_edge_deadband=args.pair_edge_deadband,
        pair_edge_strong_ic=args.pair_edge_strong_ic,
        pair_edge_very_strong_ic=args.pair_edge_very_strong_ic,
        pair_edge_weak_cap=args.pair_edge_weak_cap,
        regime_mode=args.regime_mode,
        regime_min_edge_ic=args.regime_min_edge_ic,
        regime_min_pair_edge_ic=args.regime_min_pair_edge_ic,
        regime_min_pair_count=args.regime_min_pair_count,
        regime_short_max_market_mom_24h=args.regime_short_max_market_mom_24h,
        regime_short_max_market_mom_72h=args.regime_short_max_market_mom_72h,
        regime_max_market_atr_pct=args.regime_max_market_atr_pct,
        short_max_mom_24h=args.short_max_mom_24h,
        short_max_mom_72h=args.short_max_mom_72h,
        long_min_mom_24h=args.long_min_mom_24h,
        max_entry_atr_pct=args.max_entry_atr_pct,
        short_max_market_mom_24h=args.short_max_market_mom_24h,
        short_max_market_mom_72h=args.short_max_market_mom_72h,
        short_max_market_ma_gap=args.short_max_market_ma_gap,
        short_exit_mom_24h=args.short_exit_mom_24h,
        short_exit_mom_72h=args.short_exit_mom_72h,
        short_exit_market_mom_24h=args.short_exit_market_mom_24h,
        short_exit_market_ma_gap=args.short_exit_market_ma_gap,
        exclude_pairs=args.exclude_pairs,
        candidate_state=args.candidate_state,
        recompute_corr=not args.no_corr_recompute,
    )
    print(json.dumps(summary, indent=2, default=str))


def cmd_rank_backtest(args):
    result = rank_portfolio.rank_backtest(
        tag=args.tag,
        venue=args.venue,
        timeframe=args.timeframe,
        data_venue=args.data_venue,
        pairs=args.pairs,
        top_k=args.top_k,
        gross_cap=args.gross_cap,
        net_cap=args.net_cap,
        single_pair_cap=args.single_pair_cap,
        risk_profile=args.risk_profile,
        n=args.n,
        start=args.start,
        end=args.end,
        side_mode=args.side_mode,
        min_abs_score_z=args.score_threshold,
        rebalance_hours=args.rebalance_hours,
        rebalance_minutes=args.rebalance_minutes,
        risk_per_trade=args.risk_per_trade,
        leverage_cap=args.leverage_cap,
        edge_mode=args.edge_mode,
        edge_lookback_hours=args.edge_lookback_hours,
        edge_min_periods=args.edge_min_periods,
        edge_deadband=args.edge_deadband,
        pair_edge_leverage=args.pair_edge_leverage,
        pair_edge_deadband=args.pair_edge_deadband,
        pair_edge_strong_ic=args.pair_edge_strong_ic,
        pair_edge_very_strong_ic=args.pair_edge_very_strong_ic,
        pair_edge_weak_cap=args.pair_edge_weak_cap,
        regime_mode=args.regime_mode,
        regime_min_edge_ic=args.regime_min_edge_ic,
        regime_min_pair_edge_ic=args.regime_min_pair_edge_ic,
        regime_min_pair_count=args.regime_min_pair_count,
        regime_short_max_market_mom_24h=args.regime_short_max_market_mom_24h,
        regime_short_max_market_mom_72h=args.regime_short_max_market_mom_72h,
        regime_max_market_atr_pct=args.regime_max_market_atr_pct,
        short_max_mom_24h=args.short_max_mom_24h,
        short_max_mom_72h=args.short_max_mom_72h,
        long_min_mom_24h=args.long_min_mom_24h,
        max_entry_atr_pct=args.max_entry_atr_pct,
        short_max_market_mom_24h=args.short_max_market_mom_24h,
        short_max_market_mom_72h=args.short_max_market_mom_72h,
        short_max_market_ma_gap=args.short_max_market_ma_gap,
        short_exit_mom_24h=args.short_exit_mom_24h,
        short_exit_mom_72h=args.short_exit_mom_72h,
        short_exit_market_mom_24h=args.short_exit_market_mom_24h,
        short_exit_market_ma_gap=args.short_exit_market_ma_gap,
        exclude_pairs=args.exclude_pairs,
        candidate_state=args.candidate_state,
        recompute_corr=not args.no_corr_recompute,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_lean_export(args):
    manifest = lean_bridge.export_project(
        rank_artifact=args.rank_artifact,
        output=args.output,
        timeframe=args.timeframe,
        data_root=args.data_root,
    )
    print(json.dumps(manifest, indent=2, default=str))


def cmd_lean_backtest(args):
    result = lean_bridge.run_lean_backtest(
        lean_project=args.lean_project,
        lean_bin=args.lean_bin,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_lean_compare(args):
    report = lean_bridge.compare_results(
        rank_artifact=args.rank_artifact,
        lean_result=args.lean_result,
        output=args.output,
        timeframe=args.timeframe,
    )
    print(json.dumps(report, indent=2, default=str))


def cmd_strategy_loop(args):
    if args.resume and not args.run_id:
        raise SystemExit("--run-id is required with --resume")
    # Codex review R1-#1 + R2 refinement: preflight the chosen agent CLI
    # BEFORE allocating run artifacts. The check branches on `opencode_mode`:
    #   * cli    — require local `opencode` / `hermes` binary
    #   * server — require `OPENCODE_URL` env (no local binary needed)
    #   * auto   — either is acceptable
    # Otherwise the failure surfaces deep inside _run_hermes_cli / opencode
    # invocation, leaving partial state and confusing log output on remote
    # 138 / minimal-CLI environments.
    import os as _os
    import shutil as _shutil
    agent_cli = (args.agent or "").strip().lower()
    opencode_mode = (getattr(args, "opencode_mode", "cli") or "cli").strip().lower()
    if agent_cli == "hermes":
        if _shutil.which("hermes") is None:
            raise SystemExit(
                "strategy-loop preflight: --agent hermes requires `hermes` on PATH. "
                "Install it or pick a different --agent."
            )
    elif agent_cli == "opencode":
        has_bin = _shutil.which("opencode") is not None
        has_url = bool(_os.environ.get("OPENCODE_URL"))
        if opencode_mode == "cli" and not has_bin:
            raise SystemExit(
                "strategy-loop preflight: --agent opencode --opencode-mode cli "
                "requires `opencode` on PATH. Install it or use --opencode-mode "
                "{server,auto} with OPENCODE_URL set."
            )
        if opencode_mode == "server" and not has_url:
            raise SystemExit(
                "strategy-loop preflight: --agent opencode --opencode-mode server "
                "requires the OPENCODE_URL environment variable."
            )
        if opencode_mode == "auto" and not (has_bin or has_url):
            raise SystemExit(
                "strategy-loop preflight: --agent opencode --opencode-mode auto "
                "requires EITHER `opencode` on PATH OR OPENCODE_URL env var."
            )
    result = strategy_loop.run_strategy_loop(
        tag=args.tag,
        venue=args.venue,
        agent=args.agent,
        model=args.model,
        risk_profile=args.risk_profile,
        timeframe=args.timeframe,
        data_venue=args.data_venue,
        evaluation_lane=args.lane,
        max_iterations=args.max_iterations,
        timerange=args.timerange,
        run_id=args.run_id,
        resume=args.resume,
        n=args.n,
        max_turns=args.max_turns,
        stale_timeout=args.stale_timeout,
        max_retries=args.max_retries,
        promote=not args.no_promote,
        candidate_type=args.candidate_type,
        opencode_mode=args.opencode_mode,
        hermes_provider=args.hermes_provider,
        hermes_toolsets=args.hermes_toolsets,
        hermes_reasoning_effort=args.hermes_reasoning_effort,
        hermes_yolo=args.hermes_yolo,
        candidate_state=args.candidate_state,
        recompute_corr=False if args.no_corr_recompute else None,
        baseline_profile=args.baseline_profile,
        eval_mode=args.eval_mode,
        score_mode=args.score_mode,
        promote_policy=args.promote_policy,
        validation_protocol=args.validation_protocol,
        search_timerange=args.search_timerange,
        validation_timerange=args.validation_timerange,
        blind_timerange=args.blind_timerange,
        verify_policy=args.verify_policy,
        pareto_size_per_axis=args.pareto_size_per_axis,
        lean_gate_mode=args.lean_gate_mode,
        lean_bin=args.lean_bin,
        lean_timeout=args.lean_timeout,
        lean_required_status=args.lean_required_status,
        lean_data_root=args.lean_data_root,
        score_lean_weight=args.score_lean_weight,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_strategy_loop_eval(args):
    result = strategy_loop.evaluate_candidate(
        args.candidate,
        tag=args.tag,
        venue=args.venue,
        risk_profile=args.risk_profile,
        timeframe=args.timeframe,
        data_venue=args.data_venue,
        evaluation_lane=args.lane,
        timerange=args.timerange,
        n=args.n,
        run_id=args.run_id,
        promote=args.promote,
        candidate_state=args.candidate_state,
        recompute_corr=False if args.no_corr_recompute else None,
        baseline_profile=args.baseline_profile,
        eval_mode=args.eval_mode,
        score_mode=args.score_mode,
        promote_policy=args.promote_policy,
        validation_protocol=args.validation_protocol,
        search_timerange=args.search_timerange,
        validation_timerange=args.validation_timerange,
        blind_timerange=args.blind_timerange,
        verify_policy=args.verify_policy,
        pareto_size_per_axis=args.pareto_size_per_axis,
        lean_gate_mode=args.lean_gate_mode,
        lean_bin=args.lean_bin,
        lean_timeout=args.lean_timeout,
        lean_required_status=args.lean_required_status,
        lean_data_root=args.lean_data_root,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_strategy_loop_replay(args):
    result = strategy_loop.replay_optimized_profile(
        tag=args.tag,
        baseline_profile=args.baseline_profile,
        venue=args.venue,
        risk_profile=args.risk_profile,
        timerange=args.timerange,
        include_freqtrade=not args.skip_freqtrade,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_strategy_loop_doctor(args):
    result = strategy_loop.doctor_strategy_loop_run(
        args.run_id,
        strict_formal=not args.no_strict_formal,
        write=not args.no_write,
    )
    print(json.dumps(result, indent=2, default=str))
    if not args.no_fail and not result.get("ok"):
        raise SystemExit(1)


def cmd_rank_sweep(args):
    gross_caps = [float(x) for x in str(args.gross_caps).split(",") if str(x).strip()]
    top_ks = [int(x) for x in str(args.top_ks).split(",") if str(x).strip()]
    side_modes = [str(x).strip() for x in str(args.side_modes).split(",") if str(x).strip()]
    score_thresholds = [float(x) for x in str(args.score_thresholds).split(",") if str(x).strip()]
    rebalance_hours_values = [int(x) for x in str(args.rebalance_hours_values).split(",") if str(x).strip()]
    rebalance_minutes_values = [int(x) for x in str(args.rebalance_minutes_values).split(",") if str(x).strip()] if args.rebalance_minutes_values else None
    result = rank_portfolio.rank_sweep(
        tag=args.tag,
        venue=args.venue,
        timeframe=args.timeframe,
        data_venue=args.data_venue,
        pairs=args.pairs,
        risk_profile=args.risk_profile,
        n=args.n,
        start=args.start,
        end=args.end,
        gross_caps=gross_caps,
        top_ks=top_ks,
        net_cap=args.net_cap,
        side_modes=side_modes,
        score_thresholds=score_thresholds,
        rebalance_hours_values=rebalance_hours_values,
        rebalance_minutes_values=rebalance_minutes_values,
        risk_per_trade=args.risk_per_trade,
        leverage_cap=args.leverage_cap,
        single_pair_cap=args.single_pair_cap,
        edge_mode=args.edge_mode,
        edge_lookback_hours=args.edge_lookback_hours,
        edge_min_periods=args.edge_min_periods,
        edge_deadband=args.edge_deadband,
        pair_edge_leverage=args.pair_edge_leverage,
        pair_edge_deadband=args.pair_edge_deadband,
        pair_edge_strong_ic=args.pair_edge_strong_ic,
        pair_edge_very_strong_ic=args.pair_edge_very_strong_ic,
        pair_edge_weak_cap=args.pair_edge_weak_cap,
        regime_mode=args.regime_mode,
        regime_min_edge_ic=args.regime_min_edge_ic,
        regime_min_pair_edge_ic=args.regime_min_pair_edge_ic,
        regime_min_pair_count=args.regime_min_pair_count,
        regime_short_max_market_mom_24h=args.regime_short_max_market_mom_24h,
        regime_short_max_market_mom_72h=args.regime_short_max_market_mom_72h,
        regime_max_market_atr_pct=args.regime_max_market_atr_pct,
        short_max_mom_24h=args.short_max_mom_24h,
        short_max_mom_72h=args.short_max_mom_72h,
        long_min_mom_24h=args.long_min_mom_24h,
        max_entry_atr_pct=args.max_entry_atr_pct,
        short_max_market_mom_24h=args.short_max_market_mom_24h,
        short_max_market_mom_72h=args.short_max_market_mom_72h,
        short_max_market_ma_gap=args.short_max_market_ma_gap,
        short_exit_mom_24h=args.short_exit_mom_24h,
        short_exit_mom_72h=args.short_exit_mom_72h,
        short_exit_market_mom_24h=args.short_exit_market_mom_24h,
        short_exit_market_ma_gap=args.short_exit_market_ma_gap,
        exclude_pairs=args.exclude_pairs,
        candidate_state=args.candidate_state,
        recompute_corr=not args.no_corr_recompute,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_hub(args):
    from agent_market.factor_hub import Client
    from agent_market.factor_hub import migrate as _migrate

    client = Client(db_path=args.db)

    if args.action == "init":
        path = client.init_db()
        print(f"Factor Hub DB initialized at {path}")
        if args.migrate:
            summary = _migrate.migrate_all(client, status="active")
            print(json.dumps(summary, indent=2, default=str))

    elif args.action == "migrate":
        client.init_db()
        if args.paths:
            summary = {"files": 0, "inserted": 0, "skipped": 0, "errors": 0, "per_file": []}
            for raw in args.paths:
                p = Path(raw)
                files = _migrate.discover_libraries([p]) if p.is_dir() else ([p] if p.is_file() else [])
                for f in files:
                    r = _migrate.migrate_file(client, f, status="active")
                    summary["files"] += 1
                    summary["inserted"] += r["inserted"]
                    summary["skipped"] += r["skipped"]
                    summary["errors"] += r["errors"]
                    summary["per_file"].append({"file": str(f),
                                                 **{k: r[k] for k in ("inserted", "skipped", "errors")}})
        else:
            summary = _migrate.migrate_all(client, status="active")
        print(json.dumps(summary, indent=2, default=str))

    elif args.action == "stats":
        client.init_db()
        print(json.dumps(client.stats(), indent=2))

    elif args.action == "serve":
        if args.db: os.environ["FACTOR_HUB_DB"] = args.db
        import uvicorn
        from agent_market.factor_hub.server import create_app
        app = create_app(db_path=args.db)
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.action == "ui":
        import subprocess
        ui_module = ROOT / "src" / "agent_market" / "factor_hub" / "ui.py"
        env = os.environ.copy()
        if args.db: env["FACTOR_HUB_DB"] = args.db
        subprocess.run(
            ["streamlit", "run", str(ui_module),
             "--server.port", str(args.port), "--server.headless", "true"],
            env=env, check=False,
        )

    elif args.action == "query":
        client.init_db()
        rows = client.query(
            status=args.status or None, category=args.category, origin=args.origin,
            ic_gt=args.ic_gt, metric_name=args.metric, limit=args.limit,
        )
        for r in rows:
            metric_val = r.get("latest_metric")
            if metric_val is None:
                metric_val = client.latest_metric(int(r["id"]), args.metric)
            metric_str = f"{metric_val:+.3f}" if isinstance(metric_val, (int, float)) else "   -  "
            print(f"#{r['id']:>4}  {args.metric}={metric_str}  {r['status']:<10} "
                  f"{r['origin']:<32}  {r['name'][:30]:<30}  {r['expression'][:60]}")

    elif args.action in ("sync", "deploy-from-json"):
        client.init_db()
        target = args.paths[0] if args.paths else None
        r = deploy.sync_to_hub(
            name=target, activate=not args.no_activate,
            deployment_name=args.deployment_name,
            notes=args.notes,
        )
        print(json.dumps(r, indent=2, default=str))


# ============================================================
# Argument parser
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog="factor_lab", description="Unified factor mining / backtest framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # data
    d = sub.add_parser("data", help="download raw data")
    d.add_argument("source", choices=["kucoin", "okx-futures", "bybit-futures", "binance-futures", "okx-universe", "funding"])
    d.add_argument("--timeframe", choices=["1m", "5m", "15m", "1h", "4h"], default="1h")
    d.add_argument("--pairs", default=None,
                   help="comma-separated pair list, e.g. BTC/USDT,ETH/USDT; use 'all' for all discoverable futures symbols")
    d.add_argument("--universe", default=None,
                   help="[okx-futures/okx-universe] universe name or manifest path: core_160, top200_dynamic, all_raw")
    d.add_argument("--full-history-start", default="2025-04-12",
                   help="[okx-universe] required listing cutoff for full-history core universe")
    d.add_argument("--top-n", type=int, default=None,
                   help="[okx-universe] optional override for selected instrument count")
    d.add_argument("--sleep", type=float, default=0.15,
                   help="[futures/kucoin] sleep seconds between API calls")
    d.add_argument("--data-dir", default=None,
                   help="[futures] override destination feather directory")
    d.add_argument("--start", default=None)
    d.add_argument("--end", default=None)
    d.add_argument("--years", type=int, default=None)
    d.add_argument("--aux-only", action="store_true",
                   help="[okx-futures] only create local mark/funding proxy files for Freqtrade backtests")
    d.set_defaults(func=cmd_data)

    # features
    f = sub.add_parser("features", help="merge engineered features")
    f.add_argument("kinds", nargs="+", choices=["all", "mtf4h", "xs", "pair", "funding", "micro", "ohlcv_micro", "microstructure"])
    f.add_argument("--data-dir", default=None,
                   help="directory containing spot OHLCV feathers (default: user_data/data/kucoin)")
    f.add_argument("--pairs", default="auto",
                   help="comma-separated pairs, 'auto' to discover from data-dir, or 'default'")
    f.add_argument("--pair-reference", default="BTC/USDT",
                   help="comma-separated reference pairs used by kind=pair (default: BTC/USDT)")
    f.add_argument("--pair-beta-window", type=int, default=72,
                   help="rolling beta window used by kind=pair")
    f.add_argument("--microstructure-parquet", default=None,
                   help="microstructure features.parquet from scripts/micro_features.py (LOB+trades mode)")
    f.add_argument("--microstructure-target", default=None,
                   help="target OHLCV feather (e.g. user_data/data/kucoin/BTC_USDT-1h.feather)")
    f.add_argument("--microstructure-symbol", default=None,
                   help="optional symbol filter (e.g. BTC-USDT) if parquet contains multiple symbols")
    f.add_argument("--microstructure-agg", choices=["mean", "median"], default="mean")
    f.add_argument("--microstructure-prefix", default="",
                   help="optional prefix for merged columns (default: none)")
    f.set_defaults(func=cmd_features)

    fr = sub.add_parser("features-restore", help="restore feather backups")
    fr.add_argument("kind", choices=["mtf", "xs", "pair", "funding", "micro", "ohlcv_micro", "microstructure"])
    fr.add_argument("--data-dir", default=None,
                    help="directory containing spot OHLCV feathers (default: user_data/data/kucoin)")
    fr.add_argument("--pairs", default="auto",
                    help="comma-separated pairs, 'auto' to discover from data-dir, or 'default'")
    fr.set_defaults(func=cmd_features_restore)

    # mine
    m = sub.add_parser("mine", help="iterative factor mining")
    m.add_argument("--tag", default="default", help="run identifier for checkpointing")
    m.add_argument("--rounds", type=int, default=50)
    m.add_argument("--top-k", type=int, default=40)
    m.add_argument("--llm-per-loop", type=int, default=6)
    m.add_argument("--py-per-loop", type=int, default=10)
    m.add_argument("--ic-gate", type=float, default=0.025)
    m.add_argument("--sign-gate", type=int, default=7)
    m.add_argument("--novelty-gate", type=float, default=0.85,
                    help="reject candidates with |Spearman rank corr| >= this "
                         "against any current survivor (1.0 disables)")
    m.add_argument("--hard-corr-gate", type=float, default=0.85,
                    help="hard diversity gate on |Spearman rank corr| to kept survivors")
    m.add_argument("--soft-corr-penalty-start", type=float, default=0.55,
                    help="start penalizing candidates above this |Spearman rank corr|")
    m.add_argument("--max-same-family-in-top40", type=int, default=8,
                    help="family quota normalized to top-40 survivor selection")
    m.add_argument("--max-same-signature", type=int, default=2,
                    help="maximum survivors with the same canonical source signature")
    m.add_argument("--checkpoint-every", type=int, default=10)
    m.add_argument("--lane", default="auto", choices=["auto", "1h", "4h", "15m_intraday", "5m_micro", "1m_micro"],
                    help="factor evaluation lane; auto follows --timeframe")
    m.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                    help="feather timeframe to load (must have data at that freq)")
    m.add_argument("--data-venue", default="kucoin", choices=MINING_DATA_VENUE_CHOICES,
                    help="data file convention for mining; futures venues read user_data/data/<venue>/futures/*-futures.feather")
    m.add_argument("--label-horizons", default=None,
                    help="comma-separated forward label horizons in bars; defaults from --lane")
    m.add_argument("--embargo-bars", type=int, default=0,
                    help="purged split embargo bars recorded in artifacts; 0 uses lane default")
    m.add_argument("--micro-data-quality", default="unknown",
                    choices=["unknown", "ohlcv_only", "spread_orderflow"],
                    help="data quality marker for micro lanes; 1m OHLCV-only is not promotion eligible")
    m.add_argument("--data-dir", default=None,
                   help="directory containing spot OHLCV feathers (default: user_data/data/kucoin)")
    m.add_argument("--pairs", default="auto",
                   help="comma-separated pairs, 'auto' to discover from data-dir, or 'default'")
    m.add_argument("--eval-mode", default="legacy", choices=["legacy", "composite", "portfolio"],
                    help="legacy=train/OOS IC; composite/portfolio=turnover+multi-window+XS")
    m.add_argument("--xs-weight", type=float, default=0.0,
                    help="[composite] weight of cross-sectional IC (0=pure TS, 1=pure XS)")
    m.add_argument("--turnover-weight", type=float, default=1.0,
                    help="[composite] cost penalty weight; 0 disables")
    m.add_argument("--stability-mode", default="min_abs",
                    choices=["min_abs", "mean", "median"],
                    help="[composite] how to aggregate multi-period IC")
    m.add_argument("--fee-rate", type=float, default=0.0008,
                    help="[composite] taker fee, default 0.08%% KuCoin")
    m.add_argument("--fee-bps", type=float, default=None,
                    help="[composite] taker fee in bps; overrides --fee-rate")
    m.add_argument("--slippage", type=float, default=0.0003,
                    help="[composite] slippage rate, default 3bps")
    m.add_argument("--slippage-bps", type=float, default=None,
                    help="[composite] slippage in bps; overrides --slippage")
    m.add_argument("--label-mode", default="forward_return",
                    choices=["forward_return", "pair_spread_btc", "pair_beta_resid_btc"],
                    help="target mode: raw forward return or pair-relative forward return")
    m.add_argument("--pair-reference", default="BTC/USDT",
                    help="[label-mode=pair_*] reference pair, default BTC/USDT")
    m.add_argument("--llm", action="store_true", help="enable LLM generation")
    m.add_argument("--llm-required", action="store_true",
                    help="[llm] fail the mining loop if a round produces no usable LLM candidate")
    m.add_argument("--llm-timeout", type=float, default=120.0)
    m.add_argument("--llm-retries", type=int, default=3)
    m.add_argument("--llm-max-tokens", type=int, default=0,
                   help="[llm] max completion tokens; 0 uses LLM_MAX_TOKENS or the reasoning-effort default")
    m.add_argument("--llm-reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high", "xhigh", "max"],
                    help="[llm] optional reasoning effort for compatible models/gateways")
    m.add_argument("--seed-file", default=None,
                    help="override default seed pool with single JSON file (for persona mining)")
    m.add_argument("--purify-mode", default="off", choices=["off", "clean", "neutralized", "blend"],
                    help="optional purification score mode; off preserves legacy behavior")
    m.add_argument("--purify-winsor", default="mad", choices=["mad", "quantile", "iqr", "none"],
                    help="[purify] cross-sectional winsorization method")
    m.add_argument("--purify-standardize", default="zscore",
                    choices=["zscore", "rank", "rank_gaussianize", "none"],
                    help="[purify] cross-sectional factor normalization")
    m.add_argument("--purify-neutralize", default="ridge", choices=["none", "ols", "ridge"],
                    help="[purify] residualize clean factor against exposure columns")
    m.add_argument("--purify-exposures", default="market,pair,volatility,liquidity,funding,mtf,micro",
                    help="[purify] comma-separated exposure groups to neutralize")
    m.add_argument("--alpha-objective", default="blend", choices=["blend", "pure_residual"],
                    help="ranking/gate objective; pure_residual gates on neutralized residual alpha")
    m.add_argument("--prompt-profile", default="default", choices=["default", "residual_alpha_v2"],
                    help="[llm] prompt profile; residual_alpha_v2 asks for orthogonal residual alpha")
    m.add_argument("--llm-min-feature-coverage", type=float, default=0.60,
                    help="[llm] minimum finite data coverage for a feature to be shown/accepted")
    m.add_argument("--llm-min-feature-rows", type=int, default=300,
                    help="[llm] minimum finite rows for a feature to be shown/accepted")
    m.add_argument("--llm-no-feature-filter", action="store_true",
                    help="[llm] disable low-coverage feature filtering")
    m.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help="persistent FactorLab cache directory")
    m.add_argument("--no-cache", action="store_true",
                    help="disable persistent panel/exposure/eval cache")
    m.add_argument("--lean-gate-every", type=int, default=0,
                    help="run post-loop LEAN gate every N mining loops; 1 means every loop, 0 disables")
    m.add_argument("--lean-gate-fail-fast", action="store_true",
                    help="stop mining immediately if a scheduled LEAN gate fails")
    m.add_argument("--lean-gate-no-force", action="store_true",
                    help="reuse an existing per-loop LEAN gate summary when present")
    m.add_argument("--lean-gate-n", type=int, default=30,
                    help="[lean gate] number of mined factors selected for rank portfolio")
    m.add_argument("--lean-gate-venue", default="auto", choices=["auto", *FUTURES_VENUE_CHOICES],
                    help="[lean gate] execution venue; auto follows futures --data-venue")
    m.add_argument("--lean-gate-data-venue", default="auto", choices=DATA_VENUE_CHOICES,
                    help="[lean gate] feature data venue; auto follows mining --data-venue")
    m.add_argument("--lean-gate-start", default="2025-12-01",
                    help="[lean gate] rank/LEAN backtest start date")
    m.add_argument("--lean-gate-end", default="2026-04-12",
                    help="[lean gate] rank/LEAN backtest end date")
    m.add_argument("--lean-gate-bin", default=os.environ.get("LEAN_BIN", "lean"),
                    help="[lean gate] LEAN CLI binary")
    m.add_argument("--lean-gate-timeout", type=int, default=0,
                    help="[lean gate] LEAN backtest timeout seconds; 0 disables timeout")
    m.add_argument("--lean-gate-data-root", default="",
                    help="[lean gate] override futures feather root for LEAN export")
    m.add_argument("--lean-gate-required-status", default="ok",
                    help="[lean gate] required lean-compare status")
    m.add_argument("--lean-gate-min-final-equity", type=float, default=mine_lean_gate.DEFAULT_MIN_FINAL_EQUITY)
    m.add_argument("--lean-gate-max-drawdown-pct", type=float, default=mine_lean_gate.DEFAULT_MAX_DRAWDOWN_PCT)
    m.add_argument("--lean-gate-min-trades", type=int, default=mine_lean_gate.DEFAULT_MIN_TRADES)
    m.add_argument("--lean-gate-rank-top-k", type=int, default=2,
                    help="[lean gate] rank portfolio active factor count")
    m.add_argument("--lean-gate-gross-cap", type=float, default=2.0)
    m.add_argument("--lean-gate-net-cap", type=float, default=2.0)
    m.add_argument("--lean-gate-single-pair-cap", type=float, default=2.0)
    m.add_argument("--lean-gate-side-mode", default="short", choices=["both", "long", "short"])
    m.add_argument("--lean-gate-score-threshold", type=float, default=1.5)
    m.add_argument("--lean-gate-rebalance-hours", type=int, default=8)
    m.add_argument("--lean-gate-rebalance-minutes", type=int, default=0,
                    help="[lean gate] wall-clock rebalance minutes; 0 uses hours")
    m.add_argument("--lean-gate-risk-per-trade", type=float, default=0.08)
    m.add_argument("--lean-gate-leverage-cap", type=float, default=5.0)
    m.add_argument("--lean-gate-recompute-corr", action="store_true",
                    help="[lean gate] recompute rank-series correlations inside rank portfolio")
    m.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    m.set_defaults(func=cmd_mine)

    me = sub.add_parser("mine-export", help="export top-N from a mining checkpoint")
    me.add_argument("--tag", required=True)
    me.add_argument("--n", type=int, default=30)
    me.add_argument("--diverse", action="store_true",
                    help="cluster by OOS rank correlation and export a low-correlation library")
    me.add_argument("--corr-gate", type=float, default=0.65,
                    help="[--diverse] abs Spearman rank-correlation cluster/selection gate")
    me.add_argument("--score-mode", default="combined", choices=["combined", "fitness", "portfolio"],
                    help="[--diverse] ranking mode; portfolio uses combined/stability/sign_agree")
    me.add_argument("--family-max", type=int, default=6,
                    help="[--diverse] maximum factors from one primary family")
    me.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                    help="[--diverse] timeframe used to recompute OOS rank series")
    me.add_argument("--lane", default="auto", choices=["auto", "1h", "4h", "15m_intraday", "5m_micro", "1m_micro"],
                    help="[--diverse] factor evaluation lane")
    me.add_argument("--data-venue", default="auto", choices=DATA_VENUE_CHOICES,
                    help="[--diverse] data file convention; auto inherits checkpoint config")
    me.add_argument("--label-horizons", default=None,
                    help="[--diverse] comma-separated label horizons in bars; auto inherits lane/checkpoint")
    me.add_argument("--data-dir", default=None,
                    help="[--diverse] directory containing spot OHLCV feathers")
    me.add_argument("--pairs", default="auto",
                    help="[--diverse] comma-separated pairs, 'auto' to discover from data-dir, or 'default'")
    me.add_argument("--eval-mode", default="legacy", choices=["legacy", "composite", "portfolio"],
                    help="[--diverse] eval window mode used to recompute OOS rank series")
    me.add_argument("--label-mode", default="forward_return",
                    choices=["forward_return", "pair_spread_btc", "pair_beta_resid_btc"],
                    help="[--diverse] target mode used to recompute OOS rank series")
    me.add_argument("--pair-reference", default="BTC/USDT",
                    help="[--diverse] reference pair for pair_* target modes")
    me.add_argument("--purify-mode", default="off", choices=["off", "clean", "neutralized", "blend"],
                    help="[--diverse] purification mode used to recompute rank series")
    me.add_argument("--purify-winsor", default="mad", choices=["mad", "quantile", "iqr", "none"])
    me.add_argument("--purify-standardize", default="zscore",
                    choices=["zscore", "rank", "rank_gaussianize", "none"])
    me.add_argument("--purify-neutralize", default="ridge", choices=["none", "ols", "ridge"])
    me.add_argument("--purify-exposures", default="market,pair,volatility,liquidity,funding,mtf,micro")
    me.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help="persistent FactorLab cache directory")
    me.add_argument("--no-cache", action="store_true",
                    help="disable persistent panel/exposure/eval cache")
    me.set_defaults(func=cmd_mine_export)

    mlg = sub.add_parser("mine-lean-gate", help="post-mining LEAN gate for a mined factor candidate pool")
    mlg.add_argument("--tag", required=True,
                     help="mining tag/checkpoint to validate")
    mlg.add_argument("--n", type=int, default=30,
                     help="number of factors to select for the rank portfolio stage")
    mlg.add_argument("--candidate-state", default=None,
                     help="freeze factor candidates from a specific mining state/export JSON instead of latest checkpoint")
    mlg.add_argument("--run-id", default=None,
                     help="stable run id for artifact paths; default uses tag + UTC timestamp")
    mlg.add_argument("--output", default=None,
                     help="output directory for candidate_state, LEAN project, comparison, and summary")
    mlg.add_argument("--rank-tag", default=None,
                     help="rank_portfolio artifact tag; default derives from tag + run-id")
    mlg.add_argument("--venue", default="okx", choices=FUTURES_VENUE_CHOICES)
    mlg.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h"])
    mlg.add_argument("--data-venue", default="auto", choices=DATA_VENUE_CHOICES,
                     help="feature panel venue; auto inherits candidate/rank defaults")
    mlg.add_argument("--pairs", default=None,
                     help="pair universe: default, auto/all, or comma-separated pairs")
    mlg.add_argument("--start", default="2025-12-01")
    mlg.add_argument("--end", default="2026-04-12")
    mlg.add_argument("--top-k", type=int, default=2)
    mlg.add_argument("--gross-cap", type=float, default=2.0)
    mlg.add_argument("--net-cap", type=float, default=2.0)
    mlg.add_argument("--single-pair-cap", type=float, default=2.0)
    mlg.add_argument("--risk-profile", default="aggressive", choices=["aggressive"])
    mlg.add_argument("--side-mode", default="short", choices=["both", "long", "short"])
    mlg.add_argument("--score-threshold", type=float, default=1.5,
                     help="minimum abs(rp_score_z) required for new entries")
    mlg.add_argument("--rebalance-hours", type=int, default=8)
    mlg.add_argument("--rebalance-minutes", type=int, default=None,
                     help="rebalance cadence in wall-clock minutes; overrides --rebalance-hours")
    mlg.add_argument("--risk-per-trade", type=float, default=0.08)
    mlg.add_argument("--leverage-cap", type=float, default=5.0)
    mlg.add_argument("--edge-mode", default="rolling_ic", choices=["off", "rolling_ic"])
    mlg.add_argument("--edge-lookback-hours", type=int, default=336)
    mlg.add_argument("--edge-min-periods", type=int, default=168)
    mlg.add_argument("--edge-deadband", type=float, default=0.005)
    mlg.add_argument("--pair-edge-deadband", type=float, default=None)
    mlg.add_argument("--pair-edge-strong-ic", type=float, default=None)
    mlg.add_argument("--pair-edge-very-strong-ic", type=float, default=None)
    mlg.add_argument("--pair-edge-weak-cap", type=float, default=None)
    mlg.add_argument("--pair-edge-leverage", dest="pair_edge_leverage", action="store_true")
    mlg.add_argument("--no-pair-edge-leverage", dest="pair_edge_leverage", action="store_false")
    mlg.add_argument("--regime-mode", default=None, choices=["off", "hq"])
    mlg.add_argument("--regime-min-edge-ic", type=float, default=None)
    mlg.add_argument("--regime-min-pair-edge-ic", type=float, default=None)
    mlg.add_argument("--regime-min-pair-count", type=int, default=None)
    mlg.add_argument("--regime-short-max-market-mom-24h", type=float, default=None)
    mlg.add_argument("--regime-short-max-market-mom-72h", type=float, default=None)
    mlg.add_argument("--regime-max-market-atr-pct", type=float, default=None)
    mlg.add_argument("--short-max-mom-24h", type=float, default=None)
    mlg.add_argument("--short-max-mom-72h", type=float, default=None)
    mlg.add_argument("--long-min-mom-24h", type=float, default=None)
    mlg.add_argument("--max-entry-atr-pct", type=float, default=None)
    mlg.add_argument("--short-max-market-mom-24h", type=float, default=None)
    mlg.add_argument("--short-max-market-mom-72h", type=float, default=None)
    mlg.add_argument("--short-max-market-ma-gap", type=float, default=None)
    mlg.add_argument("--short-exit-mom-24h", type=float, default=None)
    mlg.add_argument("--short-exit-mom-72h", type=float, default=None)
    mlg.add_argument("--short-exit-market-mom-24h", type=float, default=None)
    mlg.add_argument("--short-exit-market-ma-gap", type=float, default=None)
    mlg.add_argument("--exclude-pairs", default=None,
                     help="comma-separated normalized pairs to block, e.g. SOL/USDT,BTC/USDT")
    mlg.add_argument("--no-corr-recompute", action="store_true",
                     help="skip rank-series recomputation for fast diagnostics")
    mlg.add_argument("--lean-bin", default=os.environ.get("LEAN_BIN", "lean"))
    mlg.add_argument("--lean-timeout", type=int, default=None)
    mlg.add_argument("--lean-data-root", default=None,
                     help="override futures feather root for LEAN export")
    mlg.add_argument("--lean-required-status", default="ok",
                     help="required lean-compare status, or comma-separated statuses such as ok,partial")
    mlg.add_argument("--min-final-equity", type=float, default=mine_lean_gate.DEFAULT_MIN_FINAL_EQUITY)
    mlg.add_argument("--max-drawdown-pct", type=float, default=mine_lean_gate.DEFAULT_MAX_DRAWDOWN_PCT)
    mlg.add_argument("--min-trades", type=int, default=mine_lean_gate.DEFAULT_MIN_TRADES)
    mlg.add_argument("--force", action="store_true",
                     help="rerun even if the output summary already exists")
    mlg.add_argument("--no-fail-on-reject", action="store_true",
                     help="return exit code 0 even when the LEAN gate fails")
    mlg.set_defaults(pair_edge_leverage=None)
    mlg.set_defaults(func=cmd_mine_lean_gate)

    # factor-report
    frep = sub.add_parser("factor-report", help="generate IC/turnover/decay diagnostics for mined factors")
    frep.add_argument("--tag", required=True)
    frep.add_argument("--n", type=int, default=200)
    frep.add_argument("--score-mode", default="portfolio", choices=["combined", "fitness", "portfolio"])
    frep.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    frep.add_argument("--label-bars", type=int, default=None,
                      help="forward-return label horizon in bars; default inherits mining tag config")
    frep.add_argument("--data-dir", default=None)
    frep.add_argument("--data-venue", default="auto", choices=MINING_DATA_VENUE_CHOICES + ["auto"])
    frep.add_argument("--pairs", default="auto")
    frep.add_argument("--label-mode", default="forward_return",
                      choices=["forward_return", "pair_spread_btc", "pair_beta_resid_btc"])
    frep.add_argument("--pair-reference", default="BTC/USDT")
    frep.add_argument("--purify-mode", default="off", choices=["off", "clean", "neutralized", "blend"])
    frep.add_argument("--purify-winsor", default="mad", choices=["mad", "quantile", "iqr", "none"])
    frep.add_argument("--purify-standardize", default="zscore",
                      choices=["zscore", "rank", "rank_gaussianize", "none"])
    frep.add_argument("--purify-neutralize", default="ridge", choices=["none", "ols", "ridge"])
    frep.add_argument("--purify-exposures", default="market,pair,volatility,liquidity,funding,mtf,micro")
    frep.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                      help="persistent FactorLab cache directory")
    frep.add_argument("--no-cache", action="store_true",
                      help="disable persistent panel/exposure/factor cache")
    frep.set_defaults(func=cmd_factor_report)

    erep = sub.add_parser("exposure-report", help="attribute long-short factor returns to exposure groups")
    erep.add_argument("--tag", required=True)
    erep.add_argument("--n", type=int, default=200)
    erep.add_argument("--score-mode", default="portfolio", choices=["combined", "fitness", "portfolio"])
    erep.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    erep.add_argument("--label-bars", type=int, default=None,
                      help="forward-return label horizon in bars; default inherits mining tag config")
    erep.add_argument("--data-dir", default=None)
    erep.add_argument("--data-venue", default="auto", choices=MINING_DATA_VENUE_CHOICES + ["auto"])
    erep.add_argument("--pairs", default="auto")
    erep.add_argument("--label-mode", default="forward_return",
                      choices=["forward_return", "pair_spread_btc", "pair_beta_resid_btc"])
    erep.add_argument("--pair-reference", default="BTC/USDT")
    erep.add_argument("--purify-mode", default="blend", choices=["off", "clean", "neutralized", "blend"])
    erep.add_argument("--purify-winsor", default="mad", choices=["mad", "quantile", "iqr", "none"])
    erep.add_argument("--purify-standardize", default="zscore",
                      choices=["zscore", "rank", "rank_gaussianize", "none"])
    erep.add_argument("--purify-neutralize", default="ridge", choices=["none", "ols", "ridge"])
    erep.add_argument("--purify-exposures", default="market,pair,volatility,liquidity,funding,mtf,micro")
    erep.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                      help="persistent FactorLab cache directory")
    erep.add_argument("--no-cache", action="store_true",
                      help="disable persistent panel/exposure/factor cache")
    erep.add_argument("--attribution-mode", default="fast", choices=["fast", "exact"],
                      help="fast samples dates/exposures; exact runs the full attribution loop")
    erep.add_argument("--attribution-max-dates", type=int, default=128,
                      help="[fast] deterministic maximum sampled dates")
    erep.add_argument("--attribution-max-exposures", type=int, default=12,
                      help="[fast] maximum exposure columns used per factor")
    erep.set_defaults(func=cmd_exposure_report)

    cch = sub.add_parser("cache", help="inspect or clear persistent FactorLab cache")
    cch.add_argument("action", choices=["stats", "clear"])
    cch.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    cch.set_defaults(func=cmd_cache)

    ma = sub.add_parser("memory-audit", help="audit factor memory coverage, snoop labels, duplicates, and tradeability")
    ma.add_argument("--path", default=None,
                    help="factor_memory.json path (default: global factor memory)")
    ma.add_argument("--write-tags", action="store_true",
                    help="write memory_status/audit_missing_fields/quality_gate labels back to the memory store")
    ma.set_defaults(func=cmd_memory_audit)

    # validate
    v = sub.add_parser("validate", help="validate a factor library")
    v.add_argument("factors", help="path to freqai_expressions_*.json")
    v.set_defaults(func=cmd_validate)

    # backtest
    b = sub.add_parser("backtest", help="walk-forward backtest")
    b.add_argument("--tag", default="default")
    b.add_argument("--train-months", type=int, default=6)
    b.add_argument("--strategy", default="ELExitATRLSCls")
    b.add_argument("--ft-config", default="user_data/config_okx_futures_backtest.json")
    b.add_argument("--model", default="lightgbm",
                    choices=["lightgbm", "xgboost", "catboost", "stacked", "ridge_classifier"],
                    help="model adapter name (must be registered in ModelRegistry)")
    b.add_argument("--anchor", default=None,
                    help="test_start anchor date YYYY-MM-DD (default = 2025-10-09)")
    b.add_argument("--num-windows", type=int, default=0,
                    help="limit walk-forward to first N windows (0 = no limit)")
    b.add_argument("--data-start", default=None,
                    help="override feasible data_start YYYY-MM-DD")
    b.add_argument("--data-end", default=None,
                    help="override feasible data_end YYYY-MM-DD")
    b.add_argument("--expressions-file", default=None,
                    help="override expressions library (default: user_data/freqai_expressions.json)")
    b.add_argument("--exit-label-period", type=int, default=0,
                    help="if >0, also train a short-horizon exit model (e.g. 6 bars); "
                         "strategy uses it for exit decisions via AGENT_EXIT_MODEL_DIR")
    b.add_argument("--datadir", default=None,
                    help="override freqtrade --datadir (useful when feathers have extra cols)")
    b.add_argument("--n-estimators", type=int, default=0,
                    help="override model n_estimators / num_boost_round (0 = default)")
    b.set_defaults(func=cmd_backtest)

    # rank portfolio — cross-pair rank ensemble + dynamic leverage
    rx = sub.add_parser("rank-export", help="export rank-portfolio factors and per-pair feather signals")
    rx.add_argument("--tag", default="gpt54_purealpha_v2_full1000_fix1")
    rx.add_argument("--n", type=int, default=50)
    rx.add_argument("--risk-profile", default="aggressive", choices=["aggressive"])
    rx.add_argument("--venue", default="okx", choices=FUTURES_VENUE_CHOICES)
    rx.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h"])
    rx.add_argument("--data-venue", default="auto", choices=DATA_VENUE_CHOICES,
                    help="feature panel venue; auto inherits lane defaults")
    rx.add_argument("--pairs", default=None,
                    help="pair universe: default, auto/all, or comma-separated pairs; default inherits mining tag config")
    rx.add_argument("--top-k", type=int, default=None)
    rx.add_argument("--gross-cap", type=float, default=None)
    rx.add_argument("--net-cap", type=float, default=None)
    rx.add_argument("--single-pair-cap", type=float, default=None)
    rx.add_argument("--side-mode", default=None, choices=["both", "long", "short"])
    rx.add_argument("--score-threshold", type=float, default=None,
                    help="minimum abs(rp_score_z) required for new entries")
    rx.add_argument("--rebalance-hours", type=int, default=None)
    rx.add_argument("--rebalance-minutes", type=int, default=None,
                    help="rebalance cadence in wall-clock minutes; overrides --rebalance-hours")
    rx.add_argument("--risk-per-trade", type=float, default=None)
    rx.add_argument("--leverage-cap", type=float, default=None)
    rx.add_argument("--edge-mode", default=None, choices=["off", "rolling_ic"],
                    help="causal score-direction adapter; rolling_ic uses past cross-sectional IC only")
    rx.add_argument("--edge-lookback-hours", type=int, default=None)
    rx.add_argument("--edge-min-periods", type=int, default=None)
    rx.add_argument("--edge-deadband", type=float, default=None)
    rx.add_argument("--pair-edge-deadband", type=float, default=None,
                    help="per-pair rolling IC deadband for leverage alignment in rolling_ic mode")
    rx.add_argument("--pair-edge-strong-ic", type=float, default=None,
                    help="per-pair IC threshold to permit leverage above 3x")
    rx.add_argument("--pair-edge-very-strong-ic", type=float, default=None,
                    help="per-pair IC threshold to permit leverage above 5x")
    rx.add_argument("--pair-edge-weak-cap", type=float, default=None,
                    help="max leverage when per-pair IC is weak or misaligned")
    rx.add_argument("--pair-edge-leverage", dest="pair_edge_leverage", action="store_true",
                    help="enable per-pair rolling-IC dynamic leverage gating")
    rx.add_argument("--no-pair-edge-leverage", dest="pair_edge_leverage", action="store_false",
                    help="disable per-pair rolling-IC dynamic leverage gating")
    rx.add_argument("--regime-mode", default=None, choices=["off", "hq"],
                    help="date-level regime gate; hq blocks entries outside high-quality windows")
    rx.add_argument("--regime-min-edge-ic", type=float, default=None,
                    help="minimum abs(global rolling rank IC) to allow entries in hq mode")
    rx.add_argument("--regime-min-pair-edge-ic", type=float, default=None,
                    help="minimum abs(pair rolling IC) for aligned pairs in hq mode")
    rx.add_argument("--regime-min-pair-count", type=int, default=None,
                    help="minimum count of pair-edge aligned pairs to allow entries in hq mode")
    rx.add_argument("--regime-short-max-market-mom-24h", type=float, default=None,
                    help="hq mode: block short entries when market 24h momentum exceeds this cap")
    rx.add_argument("--regime-short-max-market-mom-72h", type=float, default=None,
                    help="hq mode: block short entries when market 72h momentum exceeds this cap")
    rx.add_argument("--regime-max-market-atr-pct", type=float, default=None,
                    help="hq mode: block entries when market ATR percent exceeds this cap")
    rx.add_argument("--short-max-mom-24h", type=float, default=None)
    rx.add_argument("--short-max-mom-72h", type=float, default=None)
    rx.add_argument("--long-min-mom-24h", type=float, default=None)
    rx.add_argument("--max-entry-atr-pct", type=float, default=None)
    rx.add_argument("--short-max-market-mom-24h", type=float, default=None)
    rx.add_argument("--short-max-market-mom-72h", type=float, default=None)
    rx.add_argument("--short-max-market-ma-gap", type=float, default=None)
    rx.add_argument("--short-exit-mom-24h", type=float, default=None)
    rx.add_argument("--short-exit-mom-72h", type=float, default=None)
    rx.add_argument("--short-exit-market-mom-24h", type=float, default=None)
    rx.add_argument("--short-exit-market-ma-gap", type=float, default=None)
    rx.add_argument("--exclude-pairs", default=None,
                    help="comma-separated normalized pairs to block, e.g. SOL/USDT,BTC/USDT")
    rx.add_argument("--candidate-state", default=None,
                    help="freeze factor candidates to a specific mining state JSON instead of latest checkpoint")
    rx.add_argument("--start", default=None, help="optional signal start YYYY-MM-DD")
    rx.add_argument("--end", default=None, help="optional signal end YYYY-MM-DD")
    rx.add_argument("--no-corr-recompute", action="store_true",
                    help="skip rank-series recomputation for fast diagnostics")
    rx.set_defaults(pair_edge_leverage=None)
    rx.set_defaults(func=cmd_rank_export)

    rb = sub.add_parser("rank-backtest", help="research backtest for rank-portfolio futures signals")
    rb.add_argument("--tag", default="gpt54_purealpha_v2_full1000_fix1")
    rb.add_argument("--venue", default="okx", choices=FUTURES_VENUE_CHOICES)
    rb.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h"])
    rb.add_argument("--data-venue", default="auto", choices=DATA_VENUE_CHOICES,
                    help="feature panel venue; auto inherits lane defaults")
    rb.add_argument("--pairs", default=None,
                    help="pair universe: default, auto/all, or comma-separated pairs; default inherits mining tag config")
    rb.add_argument("--top-k", type=int, default=2)
    rb.add_argument("--gross-cap", type=float, default=2.0)
    rb.add_argument("--net-cap", type=float, default=2.0)
    rb.add_argument("--single-pair-cap", type=float, default=2.0)
    rb.add_argument("--risk-profile", default="aggressive", choices=["aggressive"])
    rb.add_argument("--n", type=int, default=50)
    rb.add_argument("--start", default="2025-12-01")
    rb.add_argument("--end", default="2026-04-12")
    rb.add_argument("--side-mode", default="short", choices=["both", "long", "short"])
    rb.add_argument("--score-threshold", type=float, default=1.5,
                    help="minimum abs(rp_score_z) required for new entries")
    rb.add_argument("--rebalance-hours", type=int, default=8)
    rb.add_argument("--rebalance-minutes", type=int, default=None,
                    help="rebalance cadence in wall-clock minutes; overrides --rebalance-hours")
    rb.add_argument("--risk-per-trade", type=float, default=0.08)
    rb.add_argument("--leverage-cap", type=float, default=5.0)
    rb.add_argument("--edge-mode", default="rolling_ic", choices=["off", "rolling_ic"],
                    help="causal score-direction adapter; rolling_ic uses past cross-sectional IC only")
    rb.add_argument("--edge-lookback-hours", type=int, default=336)
    rb.add_argument("--edge-min-periods", type=int, default=168)
    rb.add_argument("--edge-deadband", type=float, default=0.005)
    rb.add_argument("--pair-edge-deadband", type=float, default=None,
                    help="per-pair rolling IC deadband for leverage alignment in rolling_ic mode")
    rb.add_argument("--pair-edge-strong-ic", type=float, default=None,
                    help="per-pair IC threshold to permit leverage above 3x")
    rb.add_argument("--pair-edge-very-strong-ic", type=float, default=None,
                    help="per-pair IC threshold to permit leverage above 5x")
    rb.add_argument("--pair-edge-weak-cap", type=float, default=None,
                    help="max leverage when per-pair IC is weak or misaligned")
    rb.add_argument("--pair-edge-leverage", dest="pair_edge_leverage", action="store_true",
                    help="enable per-pair rolling-IC dynamic leverage gating")
    rb.add_argument("--no-pair-edge-leverage", dest="pair_edge_leverage", action="store_false",
                    help="disable per-pair rolling-IC dynamic leverage gating")
    rb.add_argument("--regime-mode", default=None, choices=["off", "hq"],
                    help="date-level regime gate; hq blocks entries outside high-quality windows")
    rb.add_argument("--regime-min-edge-ic", type=float, default=None,
                    help="minimum abs(global rolling rank IC) to allow entries in hq mode")
    rb.add_argument("--regime-min-pair-edge-ic", type=float, default=None,
                    help="minimum abs(pair rolling IC) for aligned pairs in hq mode")
    rb.add_argument("--regime-min-pair-count", type=int, default=None,
                    help="minimum count of pair-edge aligned pairs to allow entries in hq mode")
    rb.add_argument("--regime-short-max-market-mom-24h", type=float, default=None,
                    help="hq mode: block short entries when market 24h momentum exceeds this cap")
    rb.add_argument("--regime-short-max-market-mom-72h", type=float, default=None,
                    help="hq mode: block short entries when market 72h momentum exceeds this cap")
    rb.add_argument("--regime-max-market-atr-pct", type=float, default=None,
                    help="hq mode: block entries when market ATR percent exceeds this cap")
    rb.add_argument("--short-max-mom-24h", type=float, default=None)
    rb.add_argument("--short-max-mom-72h", type=float, default=None)
    rb.add_argument("--long-min-mom-24h", type=float, default=None)
    rb.add_argument("--max-entry-atr-pct", type=float, default=None)
    rb.add_argument("--short-max-market-mom-24h", type=float, default=None)
    rb.add_argument("--short-max-market-mom-72h", type=float, default=None)
    rb.add_argument("--short-max-market-ma-gap", type=float, default=None)
    rb.add_argument("--short-exit-mom-24h", type=float, default=None)
    rb.add_argument("--short-exit-mom-72h", type=float, default=None)
    rb.add_argument("--short-exit-market-mom-24h", type=float, default=None)
    rb.add_argument("--short-exit-market-ma-gap", type=float, default=None)
    rb.add_argument("--exclude-pairs", default=None,
                    help="comma-separated normalized pairs to block, e.g. SOL/USDT,BTC/USDT")
    rb.add_argument("--candidate-state", default=None,
                    help="freeze factor candidates to a specific mining state JSON instead of latest checkpoint")
    rb.add_argument("--no-corr-recompute", action="store_true",
                    help="skip rank-series recomputation for fast diagnostics")
    rb.set_defaults(pair_edge_leverage=None)
    rb.set_defaults(func=cmd_rank_backtest)

    le = sub.add_parser("lean-export", help="export a local LEAN validation project from rank signals")
    le.add_argument("--rank-artifact", required=True,
                    help="rank_export.json or backtest.json produced by rank-export/rank-backtest")
    le.add_argument("--output", required=True,
                    help="output LEAN project directory, e.g. artifacts/lean/<run_id>")
    le.add_argument("--timeframe", default=None, choices=["1m", "5m", "15m", "1h", "4h"],
                    help="optional guard; must match the artifact timeframe")
    le.add_argument("--data-root", default=None,
                    help="override futures feather root for the artifact venue")
    le.set_defaults(func=cmd_lean_export)

    lb = sub.add_parser("lean-backtest", help="run local LEAN backtest for an exported bridge project")
    lb.add_argument("--lean-project", required=True,
                    help="exported LEAN project directory")
    lb.add_argument("--lean-bin", default="lean",
                    help="LEAN CLI binary/path (default: lean)")
    lb.add_argument("--timeout", type=int, default=None,
                    help="optional subprocess timeout in seconds")
    lb.set_defaults(func=cmd_lean_backtest)

    lc = sub.add_parser("lean-compare", help="compare rank research metrics with a LEAN result JSON")
    lc.add_argument("--rank-artifact", required=True,
                    help="rank_export.json or backtest.json used for the LEAN export")
    lc.add_argument("--lean-result", required=True,
                    help="LEAN result JSON or exported LEAN project directory")
    lc.add_argument("--output", default=None,
                    help="comparison JSON path (default: <lean-project>/comparison.json when discoverable)")
    lc.add_argument("--timeframe", default=None, choices=["1m", "5m", "15m", "1h", "4h"],
                    help="optional guard; must match the artifact timeframe")
    lc.set_defaults(func=cmd_lean_compare)

    sl = sub.add_parser("strategy-loop", help="agentic rank/factor strategy loop with checkpoint/resume")
    sl.add_argument("--tag", default="gpt54_purealpha_v2_full1000_fix1")
    sl.add_argument("--venue", default="okx", choices=FUTURES_VENUE_CHOICES)
    sl.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h"])
    sl.add_argument("--lane", default="auto", choices=["auto", "1h", "4h", "15m_intraday", "5m_micro", "1m_micro"])
    sl.add_argument("--data-venue", default="auto", choices=DATA_VENUE_CHOICES)
    sl.add_argument("--agent", default="hermes", choices=["hermes", "opencode"],
                    help="candidate-generation agent; Hermes is the default, OpenCode is legacy")
    sl.add_argument("--model", default=(os.environ.get("HERMES_MODEL") or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("OPENCODE_MODEL") or ""))
    sl.add_argument("--risk-profile", default="aggressive", choices=["aggressive"])
    sl.add_argument("--max-iterations", type=int, default=30)
    sl.add_argument("--timerange", default="20251201-20260412",
                    help="holdout window as YYYYMMDD-YYYYMMDD")
    sl.add_argument("--run-id", default=None, help="existing run id for --resume, or explicit id for a new run")
    sl.add_argument("--resume", action="store_true", help="resume from checkpoint.json without repeating completed phases")
    sl.add_argument("--n", type=int, default=50, help="factor count for rank-profile candidates")
    sl.add_argument("--max-turns", type=int, default=30)
    sl.add_argument("--stale-timeout", type=float, default=180.0)
    sl.add_argument("--max-retries", type=int, default=2)
    sl.add_argument("--candidate-type", default="rank_profile",
                    choices=["auto", "rank_profile", "freqtrade_strategy"],
                    help="force the agent to generate rank params or a Freqtrade strategy")
    sl.add_argument("--opencode-mode", default="cli", choices=["server", "cli", "auto"],
                    help="legacy OpenCode only: server uses runner_fsm OpenCode server; cli uses direct `opencode run`; auto falls back to cli")
    sl.add_argument("--hermes-provider", default=os.environ.get("HERMES_PROVIDER", ""),
                    help="Hermes provider override, e.g. openai-codex/openrouter/nous; default lets Hermes auto-select")
    sl.add_argument("--hermes-toolsets", default=os.environ.get("HERMES_TOOLSETS", "terminal,file"),
                    help="comma-separated Hermes toolsets for candidate generation")
    sl.add_argument("--hermes-reasoning-effort", default=os.environ.get("HERMES_REASONING_EFFORT", ""),
                    choices=["", "none", "minimal", "low", "medium", "high", "xhigh"],
                    help="Hermes agent.reasoning_effort override for compatible models")
    sl.add_argument("--hermes-yolo", action="store_true",
                    help="pass --yolo to Hermes for fully non-interactive tool execution")
    sl.add_argument("--candidate-state", default=None,
                    help="freeze factor candidates to a specific mining state JSON instead of latest checkpoint")
    sl.add_argument("--no-corr-recompute", action="store_true",
                    help="skip rank-series recomputation; default inherits optimized_profile.json when available")
    sl.add_argument("--baseline-profile", default=None,
                    help="optimized_profile.json to use as the baseline/default rank profile")
    sl.add_argument("--eval-mode", default="two_stage", choices=["research", "two_stage", "freqtrade"],
                    help="research only, two-stage research->fixed Freqtrade validation, or force Freqtrade stage")
    sl.add_argument("--score-mode", default="composite", choices=["research", "freqtrade", "composite"],
                    help="promotion/leaderboard score source; composite uses research gates and Freqtrade ranking")
    sl.add_argument("--promote-policy", default="immediate", choices=["immediate", "final", "none"],
                    help="immediate writes global winners as they appear; final writes only the run-local best at completion")
    sl.add_argument("--validation-protocol", default="single", choices=["single", "triple_holdout", "walkforward"],
                    help="single preserves legacy behavior; triple_holdout separates search/validation/blind windows")
    sl.add_argument("--search-timerange", default="20251201-20260228",
                    help="triple_holdout search window as YYYYMMDD-YYYYMMDD")
    sl.add_argument("--validation-timerange", default="20260301-20260331",
                    help="triple_holdout validation leaderboard window as YYYYMMDD-YYYYMMDD")
    sl.add_argument("--blind-timerange", default="20260401-20260412",
                    help="triple_holdout final blind window as YYYYMMDD-YYYYMMDD")
    sl.add_argument("--verify-policy", default="none", choices=["pareto", "best", "all", "none"],
                    help="which candidates get lookahead/recursive verification; triple_holdout promotion requires passed gates")
    sl.add_argument("--pareto-size-per-axis", type=int, default=3,
                    help="number of deduped candidates retained per Pareto axis")
    sl.add_argument("--lean-gate-mode", default="off", choices=["off", "final", "pareto", "all"],
                    help="run local LEAN validation as a promotion gate; enabled modes fail closed")
    sl.add_argument("--lean-bin", default=os.environ.get("LEAN_BIN", "lean"),
                    help="LEAN CLI binary/path for --lean-gate-mode")
    sl.add_argument("--lean-timeout", type=int, default=None,
                    help="optional LEAN backtest timeout in seconds")
    sl.add_argument("--lean-required-status", default="ok",
                    help="required lean-compare status, or comma-separated statuses such as ok,partial")
    sl.add_argument("--lean-data-root", default=None,
                    help="override futures feather root for LEAN export")
    sl.add_argument("--score-lean-weight", type=float, default=0.7,
                    help="weight of LEAN score in blended score (0.0=rank-only, 1.0=lean-only, default 0.7)")
    sl.add_argument("--no-promote", action="store_true",
                    help="score candidates but do not write optimized_profile.json or strategy files")
    sl.set_defaults(func=cmd_strategy_loop)

    sle = sub.add_parser("strategy-loop-eval", help="validate/backtest/score one strategy-loop candidate")
    sle.add_argument("--candidate", required=True, help="path to candidate.json")
    sle.add_argument("--tag", default="gpt54_purealpha_v2_full1000_fix1")
    sle.add_argument("--venue", default="okx", choices=FUTURES_VENUE_CHOICES)
    sle.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h"])
    sle.add_argument("--lane", default="auto", choices=["auto", "1h", "4h", "15m_intraday", "5m_micro", "1m_micro"])
    sle.add_argument("--data-venue", default="auto", choices=DATA_VENUE_CHOICES)
    sle.add_argument("--risk-profile", default="aggressive", choices=["aggressive"])
    sle.add_argument("--timerange", default="20251201-20260412",
                     help="holdout window as YYYYMMDD-YYYYMMDD")
    sle.add_argument("--n", type=int, default=50, help="factor count for rank-profile candidates")
    sle.add_argument("--run-id", default=None)
    sle.add_argument("--candidate-state", default=None,
                     help="freeze factor candidates to a specific mining state JSON instead of latest checkpoint")
    sle.add_argument("--no-corr-recompute", action="store_true",
                     help="skip rank-series recomputation; default inherits optimized_profile.json when available")
    sle.add_argument("--baseline-profile", default=None,
                     help="optimized_profile.json to use as the baseline/default rank profile")
    sle.add_argument("--eval-mode", default="research", choices=["research", "two_stage", "freqtrade"])
    sle.add_argument("--score-mode", default="research", choices=["research", "freqtrade", "composite"])
    sle.add_argument("--promote-policy", default="immediate", choices=["immediate", "final", "none"])
    sle.add_argument("--validation-protocol", default="single", choices=["single", "triple_holdout", "walkforward"])
    sle.add_argument("--search-timerange", default="20251201-20260228")
    sle.add_argument("--validation-timerange", default="20260301-20260331")
    sle.add_argument("--blind-timerange", default="20260401-20260412")
    sle.add_argument("--verify-policy", default="none", choices=["pareto", "best", "all", "none"])
    sle.add_argument("--pareto-size-per-axis", type=int, default=3)
    sle.add_argument("--lean-gate-mode", default="off", choices=["off", "final", "pareto", "all"])
    sle.add_argument("--lean-bin", default=os.environ.get("LEAN_BIN", "lean"))
    sle.add_argument("--lean-timeout", type=int, default=None)
    sle.add_argument("--lean-required-status", default="ok")
    sle.add_argument("--lean-data-root", default=None)
    sle.add_argument("--promote", action="store_true",
                     help="allow formal promotion if the candidate passes full holdout gates")
    sle.set_defaults(func=cmd_strategy_loop_eval)

    slr = sub.add_parser("strategy-loop-replay", help="replay optimized_profile.json research and fixed Freqtrade baselines")
    slr.add_argument("--tag", default="gpt54_purealpha_v2_full1000_fix1")
    slr.add_argument("--venue", default="okx", choices=FUTURES_VENUE_CHOICES)
    slr.add_argument("--risk-profile", default="aggressive", choices=["aggressive"])
    slr.add_argument("--timerange", default="20251201-20260412",
                     help="holdout window as YYYYMMDD-YYYYMMDD")
    slr.add_argument("--baseline-profile", default=None,
                     help="optimized_profile.json to replay; defaults to artifacts/rank_portfolio/<tag>/optimized_profile.json")
    slr.add_argument("--skip-freqtrade", action="store_true",
                     help="only replay the research rank_backtest stage")
    slr.set_defaults(func=cmd_strategy_loop_replay)

    sld = sub.add_parser("strategy-loop-doctor", help="audit a strategy-loop run for formal promotion readiness")
    sld.add_argument("--run-id", required=True, help="factor_strategy_loop run id")
    sld.add_argument("--no-strict-formal", action="store_true",
                     help="do not require triple_holdout + verify_policy=pareto + promote_policy=final")
    sld.add_argument("--no-write", action="store_true",
                     help="do not write doctor_latest.json into the run directory")
    sld.add_argument("--no-fail", action="store_true",
                     help="always exit 0 even when the doctor finds blockers")
    sld.set_defaults(func=cmd_strategy_loop_doctor)

    rs = sub.add_parser("rank-sweep", help="sweep rank-portfolio top-k and gross-cap settings")
    rs.add_argument("--tag", default="gpt54_purealpha_v2_full1000_fix1")
    rs.add_argument("--venue", default="okx", choices=FUTURES_VENUE_CHOICES)
    rs.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h"])
    rs.add_argument("--data-venue", default="auto", choices=DATA_VENUE_CHOICES,
                    help="feature panel venue; auto inherits lane defaults")
    rs.add_argument("--pairs", default=None,
                    help="pair universe: default, auto/all, or comma-separated pairs; default inherits mining tag config")
    rs.add_argument("--risk-profile", default="aggressive", choices=["aggressive"])
    rs.add_argument("--n", type=int, default=50)
    rs.add_argument("--start", default="2025-12-01")
    rs.add_argument("--end", default="2026-04-12")
    rs.add_argument("--gross-caps", default="1,2,3")
    rs.add_argument("--top-ks", default="1,2,3")
    rs.add_argument("--side-modes", default="short")
    rs.add_argument("--score-thresholds", default="1.5,2.0")
    rs.add_argument("--rebalance-hours-values", default="8,12,24")
    rs.add_argument("--rebalance-minutes-values", default=None,
                    help="comma-separated rebalance cadences in wall-clock minutes; overrides hours grid")
    rs.add_argument("--risk-per-trade", type=float, default=0.08)
    rs.add_argument("--leverage-cap", type=float, default=5.0)
    rs.add_argument("--single-pair-cap", type=float, default=None)
    rs.add_argument("--net-cap", type=float, default=None)
    rs.add_argument("--edge-mode", default="rolling_ic", choices=["off", "rolling_ic"])
    rs.add_argument("--edge-lookback-hours", type=int, default=336)
    rs.add_argument("--edge-min-periods", type=int, default=168)
    rs.add_argument("--edge-deadband", type=float, default=0.005)
    rs.add_argument("--pair-edge-deadband", type=float, default=None,
                    help="per-pair rolling IC deadband for leverage alignment in rolling_ic mode")
    rs.add_argument("--pair-edge-strong-ic", type=float, default=None,
                    help="per-pair IC threshold to permit leverage above 3x")
    rs.add_argument("--pair-edge-very-strong-ic", type=float, default=None,
                    help="per-pair IC threshold to permit leverage above 5x")
    rs.add_argument("--pair-edge-weak-cap", type=float, default=None,
                    help="max leverage when per-pair IC is weak or misaligned")
    rs.add_argument("--pair-edge-leverage", dest="pair_edge_leverage", action="store_true",
                    help="enable per-pair rolling-IC dynamic leverage gating")
    rs.add_argument("--no-pair-edge-leverage", dest="pair_edge_leverage", action="store_false",
                    help="disable per-pair rolling-IC dynamic leverage gating")
    rs.add_argument("--regime-mode", default=None, choices=["off", "hq"],
                    help="date-level regime gate; hq blocks entries outside high-quality windows")
    rs.add_argument("--regime-min-edge-ic", type=float, default=None,
                    help="minimum abs(global rolling rank IC) to allow entries in hq mode")
    rs.add_argument("--regime-min-pair-edge-ic", type=float, default=None,
                    help="minimum abs(pair rolling IC) for aligned pairs in hq mode")
    rs.add_argument("--regime-min-pair-count", type=int, default=None,
                    help="minimum count of pair-edge aligned pairs to allow entries in hq mode")
    rs.add_argument("--regime-short-max-market-mom-24h", type=float, default=None,
                    help="hq mode: block short entries when market 24h momentum exceeds this cap")
    rs.add_argument("--regime-short-max-market-mom-72h", type=float, default=None,
                    help="hq mode: block short entries when market 72h momentum exceeds this cap")
    rs.add_argument("--regime-max-market-atr-pct", type=float, default=None,
                    help="hq mode: block entries when market ATR percent exceeds this cap")
    rs.add_argument("--short-max-mom-24h", type=float, default=None)
    rs.add_argument("--short-max-mom-72h", type=float, default=None)
    rs.add_argument("--long-min-mom-24h", type=float, default=None)
    rs.add_argument("--max-entry-atr-pct", type=float, default=None)
    rs.add_argument("--short-max-market-mom-24h", type=float, default=None)
    rs.add_argument("--short-max-market-mom-72h", type=float, default=None)
    rs.add_argument("--short-max-market-ma-gap", type=float, default=None)
    rs.add_argument("--short-exit-mom-24h", type=float, default=None)
    rs.add_argument("--short-exit-mom-72h", type=float, default=None)
    rs.add_argument("--short-exit-market-mom-24h", type=float, default=None)
    rs.add_argument("--short-exit-market-ma-gap", type=float, default=None)
    rs.add_argument("--exclude-pairs", default=None,
                    help="comma-separated normalized pairs to block, e.g. SOL/USDT,BTC/USDT")
    rs.add_argument("--candidate-state", default=None,
                    help="freeze factor candidates to a specific mining state JSON instead of latest checkpoint")
    rs.add_argument("--no-corr-recompute", action="store_true",
                    help="skip rank-series recomputation for fast diagnostics")
    rs.set_defaults(pair_edge_leverage=None)
    rs.set_defaults(func=cmd_rank_sweep)

    # rl — PPO training + OOS rollout evaluation
    r = sub.add_parser("rl", help="train PPO agent / rollout OOS evaluation")
    r.add_argument("action", choices=["train", "eval", "bc-pretrain", "bc-eval"],
                    help="train PPO / eval PPO / pretrain BC / eval BC policy")
    r.add_argument("--tag", default="ppo_gfactors")
    r.add_argument("--expressions", default="user_data/freqai_expressions.json",
                    help="factor library to use as state features")
    r.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    r.add_argument("--timesteps", type=int, default=50_000,
                    help="[train only] total PPO timesteps")
    r.add_argument("--window-size", type=int, default=0,
                    help=">0 appends OHLCV window (activates CNN feature extractor)")
    r.add_argument("--timerange-start", default=None, help="[eval only] YYYY-MM-DD")
    r.add_argument("--timerange-end", default=None, help="[eval only] YYYY-MM-DD")
    r.add_argument("--reward-profile", default="default",
                    choices=["default", "strict"],
                    help="reward preset; strict raises fees, lowers hold penalty")
    r.add_argument("--env-class", default="trading",
                    choices=["trading", "target_position", "threshold", "trade"],
                    help="trading=buy/sell/hold; target_position={-1,0,+1}; threshold=阈值选择; trade=trade-level 自动止盈止损")
    r.add_argument("--algo-class", default="ppo", choices=["ppo", "recurrent_ppo"],
                    help="recurrent_ppo = sb3-contrib RecurrentPPO with LSTM policy")
    r.add_argument("--policy", default="MlpPolicy",
                    help="MlpPolicy / MlpLstmPolicy (auto-set for recurrent_ppo)")
    r.add_argument("--pairs", default=None,
                    help="comma-separated pair restriction for curriculum (e.g. 'BTC/USDT')")
    r.add_argument("--bc-epochs", type=int, default=5,
                    help="[bc-pretrain] supervised CE epochs on teacher labels")
    r.set_defaults(func=cmd_rl)

    # combo — GA-based combinatorial discovery
    cb = sub.add_parser("combo",
                        help="GA搜索最优因子组合（Ridge-fit linear combo, walk-forward-lite IC fitness）")
    cb.add_argument("--tag", default="combo_ga")
    cb.add_argument("--combo-size", type=int, default=13,
                    help="factors per combo (match g-factors-13)")
    cb.add_argument("--pool-size", type=int, default=300,
                    help="max candidate factors pulled from Hub")
    cb.add_argument("--min-abs-ic", type=float, default=0.08,
                    help="minimum |IC| to include a factor in the pool")
    cb.add_argument("--population", type=int, default=30)
    cb.add_argument("--generations", type=int, default=50)
    cb.add_argument("--jaccard-gate", type=float, default=0.7,
                    help="reject offspring that share >= this fraction of members with an existing combo")
    cb.add_argument("--dedupe-gate", type=float, default=0.9,
                    help="drop candidates whose |Spearman rank corr| vs a kept one >= gate (BTC ref)")
    cb.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    cb.add_argument("--include-snooped", action="store_true",
                    help="legacy mode: include factors whose selection used OOS (snoop_level != clean)")
    cb.add_argument("--seed", type=int, default=42)
    cb.add_argument("--top-n", type=int, default=3,
                    help="export this many top combos as freqai_expressions_<tag>_topN.json")
    cb.set_defaults(func=cmd_combo)

    # deploy
    dp = sub.add_parser("deploy", help="manage factor library deployment")
    dp.add_argument("action", choices=["list", "current", "switch", "describe"])
    dp.add_argument("name", nargs="?", default=None)
    dp.set_defaults(func=cmd_deploy)

    # hub — Factor Hub registry / API / dashboard
    h = sub.add_parser("hub", help="Factor Hub registry / API / dashboard")
    h.add_argument("action",
                   choices=["init", "migrate", "stats", "serve", "ui",
                            "query", "sync", "deploy-from-json"])
    h.add_argument("paths", nargs="*",
                   help="files or dirs (for migrate / sync / deploy-from-json)")
    h.add_argument("--db", default=None, help="override FACTOR_HUB_DB path")
    h.add_argument("--migrate", action="store_true",
                   help="auto-migrate known JSON libraries after init")
    h.add_argument("--host", default="127.0.0.1")
    h.add_argument("--port", type=int, default=8765)
    # query filters
    h.add_argument("--status", default="active")
    h.add_argument("--category", default=None)
    h.add_argument("--origin", default=None)
    h.add_argument("--metric", default="oos_ic")
    h.add_argument("--ic-gt", dest="ic_gt", type=float, default=None)
    h.add_argument("--limit", type=int, default=50)
    # sync / deploy-from-json
    h.add_argument("--deployment-name", default="production")
    h.add_argument("--notes", default="")
    h.add_argument("--no-activate", action="store_true",
                   help="register deployment but don't activate it")
    h.set_defaults(func=cmd_hub)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

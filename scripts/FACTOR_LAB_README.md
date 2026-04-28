# Factor Lab — Unified CLI

**One tool to rule them all**: data / features / mining / validation / backtest / deploy.

Replaces 29 standalone scripts (now in `scripts/legacy/`).

## Quick start

```bash
# From project root
python scripts/factor_lab.py --help
```

## Command reference

### 1. Data (one-time, ~2-4 hours total for full bootstrap)

```bash
# KuCoin spot OHLCV (3 years: ~1h=26k, ~4h=6.5k, ~1m=1.58M bars per pair)
python scripts/factor_lab.py data kucoin --timeframe 1h --years 3
python scripts/factor_lab.py data kucoin --timeframe 4h --years 3
python scripts/factor_lab.py data kucoin --timeframe 1m --years 3  # ~2hr, 1.5GB

# OKX SWAP futures (for backtest, 1 year)
python scripts/factor_lab.py data okx-futures

# Gate.io funding rate (3 years)
python scripts/factor_lab.py data funding
```

Output: `user_data/data/{kucoin,okx/futures,funding}/*.feather`

### 2. Features (aggregate to 1h timeline, causal)

```bash
# Merge all engineered feature types (incl. OHLCV micro_feature columns)
python scripts/factor_lab.py features all

# Or individually
python scripts/factor_lab.py features mtf4h   # 12 cols from 4h aggregation
python scripts/factor_lab.py features xs      # 7 cross-sectional ranks
python scripts/factor_lab.py features funding # 5 funding-rate derived
python scripts/factor_lab.py features micro   # 10 microstructure from 1m bars
python scripts/factor_lab.py features ohlcv_micro  # 15 OHLCV micro_feature cols (ret_1/rv_*/amihud_*)

# Restore from backup (if something went wrong)
python scripts/factor_lab.py features-restore mtf
python scripts/factor_lab.py features-restore micro
python scripts/factor_lab.py features-restore ohlcv_micro
```

Each merge writes backup at `user_data/data/kucoin/*.pre_<kind>.bak`.

#### Optional: LOB+trades microstructure (E类特征) → merge into 1h feather

This path generates columns like:
`mid/spread/rel_spread/microprice`,
`depth_bid_20/depth_ask_20/imbalance_20/slope_bid_20/convexity_20`,
`trade_sign/vwap_10/ofi_10/arrival_intensity_10/buy_vol_10/sell_vol_10`,
`expected_slippage_proxy/fill_prob_proxy/toxicity_proxy`,
`l2_ofi_tick/l2_ofi_10` (best-effort).

```bash
# 1) Capture KuCoin REST data (writes match.ndjson.gz + level2.ndjson.gz + snapshot.json)
python scripts/micro_capture.py --exchange kucoin --symbols BTC-USDT --duration-sec 600 --mode rest

# 2) Rebuild LOB state from capture output (writes lob_state.parquet)
python scripts/lob_rebuild.py \
  --capture-dir user_data/micro_capture/kucoin/<date>/<session_id> \
  --snapshot    user_data/micro_capture/kucoin/<date>/<session_id>/snapshot.json \
  --symbol      BTC-USDT \
  --out-dir     user_data/micro_capture/kucoin/<date>/<session_id>

# 3) Generate microstructure features parquet
python scripts/micro_features.py \
  --lob-state user_data/micro_capture/kucoin/<date>/<session_id>/lob_state.parquet \
  --match     user_data/micro_capture/kucoin/<date>/<session_id>/match.ndjson.gz

# 4) Merge into a target 1h feather (so miners / expression engine can use the columns)
python scripts/factor_lab.py features microstructure \
  --microstructure-parquet artifacts/runs/<run_id>/micro_feature/features.parquet \
  --microstructure-target  user_data/data/kucoin/BTC_USDT-1h.feather \
  --microstructure-symbol  BTC-USDT
```

### 3. Mine factors

```bash
# Python-only, fast (50 rounds ~ 5 min on 3-year data)
python scripts/factor_lab.py mine --tag exp_py --rounds 50

# With LLM (GPT-5.4 via cli-proxy-api, 200 rounds ~1 hour)
python scripts/factor_lab.py mine --tag exp_llm --rounds 200 --llm

# Resume from checkpoint
python scripts/factor_lab.py mine --tag exp_llm --rounds 500  # auto-resumes from loop N

# Tunable thresholds
python scripts/factor_lab.py mine --tag exp --rounds 100 \
    --ic-gate 0.03 --sign-gate 8 --top-k 50

# Export top-30 from a mining run to deployable JSON
python scripts/factor_lab.py mine-export --tag exp_py --n 30
# → writes user_data/freqai_expressions_exp_py.json
```

State: `artifacts/factor_lab/mining/<tag>/latest.json` (resume-safe).

### 4. Validate a factor library

```bash
python scripts/factor_lab.py validate user_data/freqai_expressions_exp_py.json
```

Checks: sub-period IC stability (3×2mo OOS), random baseline lift, sign consistency.

### 5. Walk-forward backtest

```bash
# 6-month training (matches g-factors baseline)
python scripts/factor_lab.py backtest --tag exp_py --train-months 6

# 24-month training (more data per window, needs 3-year data)
python scripts/factor_lab.py backtest --tag exp_py --train-months 24

# Custom strategy / freqtrade config
python scripts/factor_lab.py backtest --tag exp \
    --strategy ELExitATRLSCls --ft-config user_data/config_okx_futures_backtest.json
```

Important: `backtest` uses the currently-deployed `freqai_expressions.json`.
Switch library first via `deploy switch`.

### 6. Deploy management

```bash
# List all available libraries (⭐ marks current)
python scripts/factor_lab.py deploy list

# Show current deployment
python scripts/factor_lab.py deploy current

# Switch to a library
python scripts/factor_lab.py deploy switch exp_py
python scripts/factor_lab.py deploy switch g13.bak              # g-factors (production default)
python scripts/factor_lab.py deploy switch freqai_expressions_v4.json  # explicit filename

# Describe a library (IC stats, origins, sample exprs)
python scripts/factor_lab.py deploy describe
python scripts/factor_lab.py deploy describe freqai_expressions_v4.json
```

## Typical workflow

```bash
# 1. One-time bootstrap
python scripts/factor_lab.py data kucoin --timeframe 1h --years 3
python scripts/factor_lab.py data kucoin --timeframe 4h --years 3
python scripts/factor_lab.py data funding
python scripts/factor_lab.py data okx-futures
python scripts/factor_lab.py features all

# 2. Mine a new batch
python scripts/factor_lab.py mine --tag myrun --rounds 100 --llm
python scripts/factor_lab.py mine-export --tag myrun --n 30

# 3. Validate
python scripts/factor_lab.py validate user_data/freqai_expressions_myrun.json

# 4. Deploy + backtest
python scripts/factor_lab.py deploy switch myrun
python scripts/factor_lab.py backtest --tag myrun --train-months 6

# 5. If worse than g-factors, revert
python scripts/factor_lab.py deploy switch g13.bak
```

## Architecture

```
scripts/factor_lab.py                    # CLI entry point (~240 lines)
src/agent_market/factor_lab/
    __init__.py
    paths.py           # canonical paths and defaults
    data.py            # KuCoin/OKX/Gate downloaders
    features.py        # mtf4h/xs/funding/micro feature merge
    mining.py          # iterative IC+LLM+composition miner
    validation.py      # sub-period + random baseline
    backtest.py        # walk-forward training + freqtrade
    deploy.py          # factor library switching

scripts/legacy/        # archived old scripts (29 files)
```

## Established baselines (OOS, 7 windows × 1mo test)

| Configuration | Cumulative OOS | Notes |
|--------------|----------------|-------|
| **g-factors (13) + 92 base** | **+6.31% ⭐** | Production default |
| no expressions | +1.81% | Baseline |
| v3 deep (30 factors) | +0.15% | Over-complex |
| v4 model-driven (20) | +0.45% | LGB gain selection |
| v4 + 24mo training | +1.81% | Better but still < g |
| v5 loop500 (IC 0.141) | +1.32% | Over-optimized |
| g + xs + funding | +1.53% | Feature dilution |
| g + micro (1m data) | +0.63% | Feature dilution |

**Takeaway**: 8 independent experiments, g-factors remains optimal in current framework.
Any added complexity hurts. Real breakthroughs likely require:
- Different execution frequency (1m/5m instead of 1h)
- On-chain data (Glassnode $29/mo)
- L2 orderbook / liquidation capture (long-running)

## Available LLM providers (for `mine --llm`)

Configured via `.env`:
- **MiniMax-M2.7** (default, sometimes returns empty content)
- **DeepSeek-v3.2** (backup, faster)
- **GPT-5.4 via cli-proxy-api** (recommended if cli-proxy running on :8317 with Codex OAuth)

Switch by editing `.env`:
```
OPENAI_BASE_URL=http://127.0.0.1:8317/v1
OPENAI_MODEL=gpt-5.4
OPENAI_API_KEY=paperfarm-local
```

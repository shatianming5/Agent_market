# QuantConnect/LEAN Hybrid Validation

Agent_market remains the primary research system for OKX USDT-SWAP data, factor mining, rank signals, lane validation, and candidate promotion. The LEAN bridge is a local validation layer only: it consumes already-generated rank signals and local OKX OHLCV feathers, then exports a self-contained LEAN project for an independent event-driven execution check.

## Scope

- v1 supports local LEAN validation only.
- v1 does not upload code or data to QuantConnect Cloud.
- v1 does not migrate factor expressions into LEAN.
- OKX local USDT-SWAP feathers remain the production data source.
- Binance, Bybit, dYdX, brokerage integrations, and live trading are future work.

## Commands

```bash
python3 scripts/factor_lab.py lean-export \
  --rank-artifact artifacts/rank_portfolio/<tag>/rank_export.json \
  --output artifacts/lean/<run_id>

python3 scripts/factor_lab.py lean-backtest \
  --lean-project artifacts/lean/<run_id> \
  --lean-bin lean

python3 scripts/factor_lab.py lean-compare \
  --rank-artifact artifacts/rank_portfolio/<tag>/backtest.json \
  --lean-result artifacts/lean/<run_id>/results.json
```

`--timeframe 1h|15m|5m|1m` can be supplied to `lean-export` and `lean-compare` as a guard. It must match the artifact timeframe.

On macOS with Colima, point `DOCKER_HOST` at the active Colima socket if Docker is not on the default socket. If Colima cannot mount the system temporary directory, set `TMPDIR` to a repo path under `/Users`, for example:

```bash
DOCKER_HOST=unix:///Users/<user>/.colima/lean-validate/docker.sock \
TMPDIR=/Users/<user>/Downloads/Agent_market/artifacts/lean/tmp \
python3 scripts/factor_lab.py lean-backtest \
  --lean-project artifacts/lean/<run_id> \
  --lean-bin .venv-lean/bin/lean
```

## Exported Project

`lean-export` writes:

- `main.py`: LEAN Python algorithm consuming bridge CSV files.
- `config.json`: local LEAN project metadata.
- `manifest.json`: artifact lineage, timeframe, risk config, pair mapping, coverage report.
- `data/signals.csv`: rank targets with LEAN-ready hold/exit/kill semantics.
- `data/ohlcv/<SYMBOL>.csv`: OKX OHLCV converted from local feathers.

The bridge exports only pairs used by the signal set and fails before writing invalid projects when OKX data is missing, duplicated, gapped, or not aligned with signal timestamps.

## Execution Semantics

LEAN consumes `lean_target_weight`, derived from `rp_target_weight`:

- `rp_rebalance=False` keeps the previous target.
- `rp_exit_long`, `rp_exit_short`, `rp_liq_reject`, and non-normal `rp_kill_mode` flatten the pair.
- Gross cap, leverage cap, fee rate, and slippage are injected from `risk_config`.

The v1 execution model is `close_to_next_bar_approx` with `order_on_lean_action_only`: LEAN submits orders only when the carried target changes or a force-flat signal fires. Between those actions LEAN holds filled quantity, so exposure can drift with price. This is the executable event-driven interpretation; it is expected to differ from a research return formula that applies target weights every bar without drift-rebalance orders.

## Comparison

`lean-compare` writes `comparison.json` beside the LEAN project when the project can be discovered. It compares:

- final equity
- total return
- max drawdown
- profit/max drawdown
- trades: entry count, i.e. flat to non-zero exposure; sign flips count as a new entry
- orders: research target-change instructions versus LEAN filled order events
- turnover: sum of absolute executed notional divided by contemporaneous equity; LEAN summary `Portfolio Turnover` is diagnostic only
- average gross
- max gross
- fee cost
- slippage cost

Default drift thresholds are 5% for final equity, 10% for max drawdown, 5% for trades, 5% for orders, and 10% for turnover. For `5m` and `1m`, the report is marked as micro validation and includes cost sensitivity output.

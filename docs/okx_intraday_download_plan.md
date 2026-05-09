# OKX Intraday Futures Download Plan

## Goal

Build a production-grade OKX USDT-SWAP OHLCV dataset for the rank factor lanes:

- `1h`: baseline rank lane.
- `15m`: first production intraday alpha lane.
- `5m`: micro lane candidate validation.
- `1m`: microstructure research lane; OHLCV-only data remains research/candidate only until spread/orderflow data is added.

## Production Universe

Primary universe is the current rank portfolio default set:

- `BTC/USDT`
- `ETH/USDT`
- `SOL/USDT`
- `BNB/USDT`
- `XRP/USDT`
- `DOGE/USDT`
- `ADA/USDT`
- `AVAX/USDT`
- `LINK/USDT`
- `DOT/USDT`

Extended research universe can be added after the primary universe is complete:

- `APT/USDT`
- `ARB/USDT`
- `NEAR/USDT`
- `OP/USDT`
- `SUI/USDT`

Large-universe expansion uses generated manifests under `user_data/data/okx/universes/`:

- `okx_core_160.json`: 160 highest-liquidity live OKX USDT-SWAP contracts listed on or before `2025-04-12`; use this for full-history production rank evaluation.
- `okx_top200_dynamic.json`: 200 highest-liquidity live OKX USDT-SWAP contracts; use this for factor mining with dynamic listing-date filters.
- `okx_all303_raw.json`: all live OKX USDT-SWAP contracts discovered from OKX; use this as a raw research/data warehouse universe only.

## Canonical Storage

Files are written under:

`user_data/data/okx/futures/`

Naming convention:

- OHLCV: `<BASE>_USDT_USDT-<timeframe>-futures.feather`
- mark proxy: `<BASE>_USDT_USDT-<timeframe>-mark.feather`
- funding proxy: `<BASE>_USDT_USDT-<timeframe>-funding_rate.feather`

The mark/funding proxy files are generated automatically after each timeframe download.

## Date Range

Canonical range for the first complete intraday dataset:

- Start: `2025-04-12`
- End: `2026-04-30`

`end` is exclusive at `00:00 UTC`. OKX history candles exclude the exact start boundary, so first rows commonly begin at `start + timeframe`.

## Execution Order

1. Finish `15m` for the primary universe.
2. Run coverage validation for `15m`.
3. Download `5m` for the primary universe.
4. Run coverage validation for `5m`.
5. Download `1m` for the primary universe.
6. Run coverage validation for `1m`.
7. Refresh `1h` to the same end date.
8. Run final all-timeframe coverage validation.
9. Only after primary universe passes, optionally download the extended research universe.

## Commands

Primary universe:

```bash
python3 scripts/factor_lab.py data okx-futures --timeframe 15m --start 2025-04-12 --end 2026-04-30 --sleep 0.02
python3 scripts/factor_lab.py data okx-futures --timeframe 5m  --start 2025-04-12 --end 2026-04-30 --sleep 0.02
python3 scripts/factor_lab.py data okx-futures --timeframe 1m  --start 2025-04-12 --end 2026-04-30 --sleep 0.02
python3 scripts/factor_lab.py data okx-futures --timeframe 1h  --start 2025-04-12 --end 2026-04-30 --sleep 0.02
```

Large-universe manifests:

```bash
python3 scripts/factor_lab.py data okx-universe --full-history-start 2025-04-12
```

Resumable large-universe downloads:

```bash
python3 scripts/okx_universe_download.py --universe core_160 --timeframes 15m,5m,1m,1h --start 2025-04-12 --end 2026-04-30 --sleep 0.02 --batch-size 10 --workers 4
python3 scripts/okx_universe_download.py --universe top200_dynamic --timeframes 15m,5m,1m,1h --start 2025-04-12 --end 2026-04-30 --sleep 0.02 --batch-size 10 --workers 4
python3 scripts/okx_universe_download.py --universe all_raw --timeframes 15m,5m,1m,1h --start 2025-04-12 --end 2026-04-30 --sleep 0.02 --batch-size 10 --workers 4
python3 scripts/okx_universe_download_queue.py --universes core_160,top200_dynamic,all_raw --timeframes 1h,15m,5m,1m --start 2025-04-12 --end 2026-04-30 --sleep 0.02 --batch-size 10 --workers 4
```

Single timeframe from a manifest:

```bash
python3 scripts/factor_lab.py data okx-futures --universe core_160 --timeframe 15m --start 2025-04-12 --end 2026-04-30 --sleep 0.02
```

Single-pair retry example:

```bash
python3 scripts/factor_lab.py data okx-futures --timeframe 5m --start 2025-04-12 --end 2026-04-30 --pairs BTC/USDT --sleep 0.02
```

Extended research universe:

```bash
python3 scripts/factor_lab.py data okx-futures --timeframe 15m --start 2025-04-12 --end 2026-04-30 --pairs APT/USDT,ARB/USDT,NEAR/USDT,OP/USDT,SUI/USDT --sleep 0.02
python3 scripts/factor_lab.py data okx-futures --timeframe 5m  --start 2025-04-12 --end 2026-04-30 --pairs APT/USDT,ARB/USDT,NEAR/USDT,OP/USDT,SUI/USDT --sleep 0.02
python3 scripts/factor_lab.py data okx-futures --timeframe 1m  --start 2025-04-12 --end 2026-04-30 --pairs APT/USDT,ARB/USDT,NEAR/USDT,OP/USDT,SUI/USDT --sleep 0.02
python3 scripts/factor_lab.py data okx-futures --timeframe 1h  --start 2025-04-12 --end 2026-04-30 --pairs APT/USDT,ARB/USDT,NEAR/USDT,OP/USDT,SUI/USDT --sleep 0.02
```

## Expected Size And Runtime

Approximate bars per pair for `2025-04-12` to `2026-04-30`:

- `15m`: about 36.7k rows.
- `5m`: about 110k rows.
- `1m`: about 552k rows.
- `1h`: about 9.2k rows.

OKX `history-candles` returns at most 100 rows per request. Based on observed `15m` speed:

- `15m` primary universe: about 25 minutes.
- `5m` primary universe: about 75 minutes.
- `1m` primary universe: several hours; run as a long resumable job.
- `1h` primary universe: a few minutes.

## Resumability

The downloader merges with existing files, dedupes by `date`, and skips files that already cover the requested range. If a run stops:

1. Re-run the same command.
2. Completed pair files are skipped.
3. Partial pair files are merged and completed.

For dynamic universes, each manifest includes `pair_start_dates`. New listings are considered complete from their own listing date, so re-runs do not repeatedly fetch pre-listing history.

## Acceptance Criteria

For each primary-universe timeframe:

- 10 futures OHLCV files exist.
- 10 mark proxy files exist.
- 10 funding proxy files exist.
- All 10 default pairs are present.
- Common overlap covers at least `2025-04-12` to `2026-04-29`.
- Gap files count is 0 for `15m`, `5m`, and `1h`.
- `1m` should have 0 gaps after completion; if OKX returns missing bars, record exact gaps before using the lane.

## Promotion Policy

- `15m` may enter production rank-loop evaluation after coverage passes.
- `5m` may enter rank-loop evaluation but must pass turnover/cost gates.
- `1m` remains research/candidate-only with `micro_data_quality=ohlcv_only` and `promotion_eligible=false` until spread/orderflow/liquidity data is available.

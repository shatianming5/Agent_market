# Validation Checklist

Use this checklist when auditing Agent_market factor/rank strategy loops.

## Local Evidence

- Active run id, checkpoint status, current iteration, phase, and config.
- Leaderboard row count and best row by selected score.
- Best row `score_components`, research metrics, Freqtrade metrics, parameter signature, and promotion record.
- Recent failed iterations and whether violations are short enough to avoid prompt/checkpoint pollution.
- Exact timerange and data directory used by research and Freqtrade stages.

## Research/Freqtrade Alignment

- Confirm research `rank_backtest` uses the same rank profile as signal export.
- Confirm Freqtrade loader reads the exported signal directory intended for that iteration.
- Confirm Freqtrade timerange matches the research window.
- Confirm pair normalization handles futures names such as `ETH/USDT:USDT`.
- Confirm stake/position sizing consumes `rp_target_weight` and direction logic consumes `rp_side`.

## Leakage Checks

- Rank features must be computed only from past or current candle data.
- Cross-sectional ranks must not use future labels or future universe membership.
- Signal export must not shift targets backward in a way that exposes future values.
- Freqtrade merge/join logic must not forward-fill future signals into earlier candles.
- Any cached factor state must be frozen and path-recorded.

## Freqtrade Bias Tools

Require a protocol for:
- `freqtrade lookahead-analysis` against the fixed rank strategy/config.
- `freqtrade recursive-analysis` against the same strategy/config.

Record exact command, config path, strategy path, timerange, signal directory, and pass/fail threshold.

## Overfitting and Selection Bias

Check for:
- Too many repeated trials on one fixed window.
- Hyperparameter precision finer than economically meaningful granularity.
- Best-of-many selection without a holdout or correction.
- Lack of walk-forward evaluation.
- Lack of out-of-time or market-regime split.

Suggested minimum protocol:
- Development window: used for search.
- Validation window: used for candidate ranking during loop.
- Final holdout: untouched until final candidate is frozen.
- Walk-forward slices: multiple contiguous periods with no parameter refit inside a slice.

## Acceptance Criteria

A candidate is robust enough to promote only when:
- Research and Freqtrade hard gates pass on the selected window.
- Lookahead analysis has no material bias flags.
- Recursive analysis differences are within documented tolerance.
- Walk-forward results do not rely on one single exceptional period.
- Final holdout passes minimum trades, max drawdown, zero liquidations, and minimum profit/drawdown.
- The claim text matches the evidence strength.

---
name: strategy-deepresearch
description: Sidecar deep-research audit workflow for Agent_market factor/rank strategy loops. Use when Codex or Hermes should investigate trading-strategy validation quality, lookahead/recursive bias, backtest overfitting, walk-forward protocol, Freqtrade/research metric alignment, or produce docs/strategy_research_review.md and docs/validation_protocol.md without changing a running loop.
---

# Strategy Deep Research

## Scope

Use this skill to audit a strategy loop from the side. Do not use it as an inner-loop candidate generator.

The audit should answer:
- Are the research backtest, rank signal export, and Freqtrade validation measuring the same strategy?
- Are lookahead bias, recursive indicator drift, data leakage, survivorship, and multiple-backtest selection bias controlled?
- Which experiments must be added before treating a leaderboard winner as robust?
- Which claims are allowed by the current evidence, and which are still unsupported?

## Non-Negotiables

- Do not stop, resume, or modify tmux strategy loops unless the user explicitly asks.
- Do not edit `optimized_profile.json`, strategy code, or loop controller code during an audit.
- Treat web/literature findings as evidence for validation design, not as direct trading advice.
- Cite sources for external claims. Prefer official docs, original papers, and primary source material.
- Keep outputs reproducible: write exact commands, artifact paths, run ids, timeranges, and data windows.

## Quick Start

From the repo root, collect local loop evidence first:

```bash
python3 skills/strategy-deepresearch/scripts/collect_strategy_context.py \
  --run-id RUN_ID \
  --out artifacts/strategy_deepresearch/RUN_ID/context.json
```

If no run id is supplied, the script chooses the most recently updated run under `artifacts/factor_strategy_loop/`.

Then perform the audit and write:
- `docs/strategy_research_review.md`
- `docs/validation_protocol.md`

Use `references/validation-checklist.md` for the audit checklist and `references/report-template.md` for the expected report shape.
Use `references/external-skill-resources.md` when the user asks to compare or borrow ideas from external Claude/Codex/Hermes skill collections.

## Audit Workflow

1. **Collect local evidence**
   - Run `collect_strategy_context.py`.
   - Read the active run `checkpoint.json`, `leaderboard.json`, best iteration artifacts, and recent failures.
   - Inspect relevant controller/loader files only as needed:
     - `src/agent_market/factor_lab/strategy_loop.py`
     - `src/agent_market/factor_lab/rank_portfolio.py`
     - `user_data/strategies/ELRankPortfolioLeverageStrategy.py`
     - `scripts/freqtrade_cli.py`

2. **Check scoring and promotion integrity**
   - Verify `score_mode`, `eval_mode`, and `promote_policy`.
   - Confirm promotion score matches the intended stage: research, Freqtrade, or composite.
   - Check whether run-local `best/` and final global promotion are separated.

3. **Audit leakage and robustness risks**
   - Look for future-data access in rank features, signal export, and Freqtrade loader joins.
   - Check whether Freqtrade `lookahead-analysis` and `recursive-analysis` are planned or already run.
   - Check whether train/validation/test windows are separated and whether walk-forward/out-of-time tests exist.
   - Check whether many repeated backtests have created selection bias.

4. **Run external research only for validation questions**
   - Use web search for current official docs and primary papers when needed.
   - For Freqtrade behavior, prefer official Freqtrade docs.
   - For overfitting/selection-bias claims, prefer original statistical finance papers or well-cited primary material.
   - Record source title, URL, date accessed, and the audit implication.
   - When reviewing third-party skill collections, mine them for workflow patterns only; do not install or run unreviewed code.

5. **Write the deliverables**
   - `docs/strategy_research_review.md`: concise findings, evidence, risk severity, and immediate actions.
   - `docs/validation_protocol.md`: exact commands and acceptance criteria for lookahead, recursive, walk-forward, and holdout tests.
   - If useful, also write `artifacts/strategy_deepresearch/<run_id>/sources.json`.

## Output Standards

Use concrete file/line references for local code findings. Use URLs for external sources.

Classify each finding:
- `BLOCKER`: invalidates leaderboard/promotion until fixed.
- `HIGH`: likely changes ranking or live reliability.
- `MEDIUM`: important robustness gap but not immediately invalidating.
- `LOW`: documentation, monitoring, or polish.

Every action item must include:
- Owner surface: controller, rank engine, Freqtrade strategy, experiment protocol, or docs.
- Exact verification command.
- Pass/fail criterion.

## Hermes Usage

Hermes can preload this skill with:

```bash
hermes -s strategy-deepresearch chat -q "Audit the active Agent_market strategy loop and write the two docs."
```

When Hermes is invoked by `strategy-loop` candidate generation, do not preload this skill. It is meant for separate audit sessions, not for per-iteration candidate writing.

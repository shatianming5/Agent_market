# Strategy Deep Research Report Template

Use this structure for `docs/strategy_research_review.md`.

```markdown
# Strategy Deep Research Review

**Run id:** ...
**Date:** ...
**Scope:** sidecar audit; no candidate generation or loop mutation

## Executive Judgment

- Overall status: BLOCKED / HIGH RISK / CONDITIONALLY OK / OK
- Main reason:
- Next required gate:

## Local Evidence

| Artifact | Path | Finding |
|---|---|---|
| checkpoint | ... | ... |
| leaderboard | ... | ... |
| best iteration | ... | ... |

## Findings

### BLOCKER-001: ...

Evidence:
- Local:
- External:

Impact:

Required fix or experiment:

Verification:
```bash
...
```

Pass criterion:

## Scoring and Promotion

- Score mode:
- Eval mode:
- Promote policy:
- Best selected metric:
- Any mismatch:

## Bias and Robustness Matrix

| Risk | Current Evidence | Gap | Required Test | Severity |
|---|---|---|---|---|
| Lookahead bias | ... | ... | ... | ... |
| Recursive drift | ... | ... | ... | ... |
| Selection bias | ... | ... | ... | ... |
| Walk-forward robustness | ... | ... | ... | ... |

## External Sources

| Source | URL | Why it matters |
|---|---|---|

## Action Plan

1. ...
2. ...
3. ...
```

Use this structure for `docs/validation_protocol.md`.

```markdown
# Strategy Validation Protocol

## Scope

## Required Artifacts

## Commands

### 1. Reproduce Best Candidate

### 2. Freqtrade Lookahead Analysis

### 3. Freqtrade Recursive Analysis

### 4. Walk-Forward Backtests

### 5. Final Holdout

## Acceptance Gates

## Evidence Log
```

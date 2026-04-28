# Handoff

- Runtime plane must stay direct OpenCode.
- Ralph loop only manages implementation backlog.
- Update progress.md and acceptance.md after each bounded pass.
- This bounded pass already landed:
  - OpenCode provider support for factor mining.
  - Direct factor-flow config for OpenCode.
  - Strategy-miner OpenCode runtime config renderer.
  - Repo-local Ralph loop scaffolds for `architecture`, `factor-mining`, and `strategy-evolution`.
  - CLI smoke checks for the new factor-mining flags and strategy runtime config renderer.
- Do not replace runtime entrypoints with a Ralph wrapper. Ralph only tracks implementation work.
- Next preferred pass: factor-card / factor-memory persistence.

# Contributing Guide

Thank you for investing time in Agent Market. This document explains how to set up your environment, follow the code style, and open effective pull requests.

## Local Environment
1. Install **Python 3.11+** and **Node.js 20+**.
2. Create a virtual environment and install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   python -m pip install -r server/requirements.txt
   python -m pip install -r requirements-dev.txt
   ```
3. Install front-end packages:
   ```powershell
   npm install --prefix web
   ```
4. （可选）启用 pre-commit 钩子：
   ```powershell
   pre-commit install
   ```

## Branch & Commit Rules
- Branch names follow `<type>/<nickname>` (examples: `feat/runtime-dag`, `fix/order-status`).
- Keep commits focused; write clear messages in the imperative mood (“Add job progress endpoint”).
- Reference issues or TODO IDs when relevant.

## Coding Standards
| Area | Tooling | Command |
| --- | --- | --- |
| Python | Ruff (lint), Black (format) | `ruff check .`, `black .` |
| TypeScript / React | ESLint, Prettier | `npm --prefix web run lint`, `npm --prefix web run format:fix` |
| All files | pre-commit hooks | `pre-commit run --all-files` |

Additional guidelines:
- Prefer type hints and docstrings for new Python modules.
- Keep React components small and rely on hooks for side effects.
- Update or add tests (pytest or Vitest/Playwright once available) when changing behaviour.

## Testing Checklist
- `pytest -q` 运行 Python 单元测试。
- `npm --prefix web run test` (placeholder) should be introduced alongside front-end tests; until then manually verify critical flows through the UI and `scripts/start_both_and_test.py`.
- For long running pipelines use `scripts/server_quickcheck.py` as a smoke test before opening a pull request.

## Documentation Expectations
- Update relevant docs under `docs/` when you modify behaviour, configuration, or external interfaces.
- Keep diagrams in Mermaid so they render consistently in Markdown and the developer portal.

## Pull Request Process
1. Rebase on the latest `main` (or the release branch you target).
2. Run `ruff check .`, `black --check .`, and `pytest -q`.
3. Attach artefacts (logs, screenshots) for notable UI or ML changes.
4. Request review from the relevant component owners.

## Reporting Issues
- Use GitHub issues for bugs or feature requests. Include environment details, reproduction steps, and expected vs actual outcomes.
- Security-related topics should follow the process in `docs/SECURITY.md`.

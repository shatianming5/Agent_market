# Review Deferred Items Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 4 deferred items from the auto-review: split phases.py, wire walk-forward evaluation, complete dependency manifest, broaden CI coverage.

**Architecture:** phases.py splits into 6 private submodules under strategy_miner/ with a thin re-export layer. Walk-forward adds an optional multi-period backtest loop in phase_evaluation. Dependencies get completed in requirements-full.txt. CI gets py_compile + import smoke + broader test coverage.

**Tech Stack:** Python 3.11+, pytest, freqtrade, lightgbm/xgboost/torch/sb3

---

## Task 1: Split phases.py — Create _helpers.py

**Files:**
- Create: `src/agent_market/strategy_miner/_helpers.py`
- Modify: `src/agent_market/strategy_miner/phases.py`

**Step 1: Create _helpers.py with shared utility functions**

Extract these functions (phases.py lines 49-534, 1170-1262) into `_helpers.py`:

```
_truncate_text            (49-57)
_freqtrade_config_defaults (58-71)
_split_timerange          (72-91)
_timeframe_to_minutes     (108-125)
_prompt_objective_profile (126-163)
_extract_strategy_timeframes (164-213)
_validate_timeframe_policy (214-235)
_CANDIDATE_TYPE_FAMILIES  (412-418)
_normalize_candidate_type (420-424)
_configured_candidate_types (425-434)
_candidate_type_for_slot  (435-446)
_allowed_model_families   (447-450)
_candidate_requires_training (451-458)
_phase_for_candidate      (459-464)
_load_freqtrade_payload   (465-472)
_freqtrade_market_context (473-481)
_sanitize_candidate_name  (482-493)
_json_block_or_none       (494-497)
_coerce_float             (498-504)
_coerce_int               (505-515)
_normalize_roi_map        (516-534)
_parse_json_object        (1170-1194)
_classify_validation_failure (1195-1213)
_pick_active_candidate    (1214-1233)
_mark_candidate_done      (1234-1245)
_advance_after_candidate  (1246-1251)
_rewrite_strategy_class_name (1252-1262)
```

Include imports needed by these functions:
```python
from __future__ import annotations
import ast, json, logging, re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
if TYPE_CHECKING:
    from .dtypes import MinerConfig, MinerState, StrategyCandidate
from agent_market import paths
from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
```

**Step 2: Update phases.py to import from _helpers**

Replace the extracted function bodies in phases.py with:
```python
from ._helpers import (
    _truncate_text, _freqtrade_config_defaults, _split_timerange,
    _timeframe_to_minutes, _prompt_objective_profile,
    _extract_strategy_timeframes, _validate_timeframe_policy,
    _CANDIDATE_TYPE_FAMILIES, _normalize_candidate_type,
    _configured_candidate_types, _candidate_type_for_slot,
    _allowed_model_families, _candidate_requires_training,
    _phase_for_candidate, _load_freqtrade_payload,
    _freqtrade_market_context, _sanitize_candidate_name,
    _json_block_or_none, _coerce_float, _coerce_int,
    _normalize_roi_map, _parse_json_object,
    _classify_validation_failure, _pick_active_candidate,
    _mark_candidate_done, _advance_after_candidate,
    _rewrite_strategy_class_name,
)
```

**Step 3: Run tests**

Run: `pytest tests/test_strategy_miner_phases.py tests/test_strategy_miner_runner.py tests/test_pipeline_leakage.py -v --timeout=30`
Expected: All pass (no behavior change)

---

## Task 2: Split phases.py — Create _scoring.py

**Files:**
- Create: `src/agent_market/strategy_miner/_scoring.py`
- Modify: `src/agent_market/strategy_miner/phases.py`

**Step 1: Create _scoring.py**

Extract these functions:
```
_compute_psr              (92-107)
_compute_effective_score  (236-315)
_training_score_adjustment (316-346)
_training_robustness_violations (347-373)
_check_per_pair_robustness (374-419)
_safe_metric              (3121-3133)
_safe_ratio_metric        (3134-3140)
```

Imports needed:
```python
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
from ._helpers import _coerce_float
from .dtypes import MinerConfig, StrategyCandidate
```

**Step 2: Update phases.py imports**

**Step 3: Run tests**

Run: `pytest tests/test_strategy_miner_phases.py -v --timeout=30`
Expected: All pass

---

## Task 3: Split phases.py — Create _rendering.py

**Files:**
- Create: `src/agent_market/strategy_miner/_rendering.py`
- Modify: `src/agent_market/strategy_miner/phases.py`

**Step 1: Create _rendering.py**

Extract these functions:
```
_sanitize_model_params      (535-584)
_normalize_model_candidate_payload (585-770)
_format_runtime_literal     (771-774)
_render_ml_strategy_code    (775-949)
_render_rl_signal_strategy_code (950-1040)
_restore_trained_wrapper    (1041-1128)
```

Imports needed:
```python
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, Optional
from agent_market import paths
from ._helpers import (
    _coerce_float, _coerce_int, _normalize_roi_map,
    _freqtrade_config_defaults, _freqtrade_market_context,
    _normalize_candidate_type, _prompt_objective_profile,
)
from .dtypes import MinerConfig, StrategyCandidate
from .sandbox import ensure_freqtrade_strategy_compliance_file
```

**Step 2: Update phases.py imports**

**Step 3: Run tests**

---

## Task 4: Split phases.py — Create _generation.py

**Files:**
- Create: `src/agent_market/strategy_miner/_generation.py`
- Modify: `src/agent_market/strategy_miner/phases.py`

**Step 1: Create _generation.py**

Extract:
```
_INDICATOR_SETS           (1113-1127)
_extract_indicator_names  (1129-1143)
_build_market_profile     (1144-1169)
_normalize_candidate_artifact (1263-1310)
phase_strategy_gen        (1497-2319)
```

This is the largest extraction (~850 lines). It needs imports from `_helpers`, `_rendering`, plus the prompt builders, sandbox, agent modules.

**Step 2: Update phases.py — remove extracted code, add import**

```python
from ._generation import phase_strategy_gen
```

**Step 3: Run tests**

---

## Task 5: Split phases.py — Create _backtest.py

**Files:**
- Create: `src/agent_market/strategy_miner/_backtest.py`
- Modify: `src/agent_market/strategy_miner/phases.py`

**Step 1: Create _backtest.py**

Extract:
```
_repair_candidate          (1311-1496)
_classify_backtest_failure (2320-2386)
_classify_train_failure    (2387-2402)
phase_train_model          (2403-2573)
phase_backtest             (2574-3120)
```

**Step 2: Update phases.py imports**

**Step 3: Run tests**

---

## Task 6: Split phases.py — Create _evaluation.py + finalize phases.py

**Files:**
- Create: `src/agent_market/strategy_miner/_evaluation.py`
- Modify: `src/agent_market/strategy_miner/phases.py`

**Step 1: Create _evaluation.py**

Extract:
```
phase_evaluation  (3141-3338)
phase_analysis    (3339-3451)
```

**Step 2: Reduce phases.py to a thin re-export module**

Final phases.py (~20 lines):
```python
"""Phase handlers for the strategy mining loop.

Split into submodules for maintainability:
  _helpers.py    — shared utilities
  _scoring.py    — metric computation
  _rendering.py  — ML/RL strategy code rendering
  _generation.py — phase_strategy_gen
  _backtest.py   — phase_train_model + phase_backtest
  _evaluation.py — phase_evaluation + phase_analysis
"""
from ._backtest import phase_backtest, phase_train_model
from ._evaluation import phase_analysis, phase_evaluation
from ._generation import phase_strategy_gen

__all__ = [
    "phase_strategy_gen",
    "phase_train_model",
    "phase_backtest",
    "phase_evaluation",
    "phase_analysis",
]
```

**Step 3: Run full test suite**

Run: `pytest tests/ -q --timeout=30 --ignore=tests/test_e2e_flow_smoke.py`
Expected: Same pass count as before (239+)

---

## Task 7: Walk-forward — Add config + timerange splitting

**Files:**
- Modify: `src/agent_market/strategy_miner/dtypes.py`
- Modify: `src/agent_market/strategy_miner/_helpers.py` (or phases.py if not yet split)

**Step 1: Add walk-forward config fields to MinerConfig**

```python
# Walk-forward OOS validation (optional)
walkforward_enabled: bool = False
walkforward_folds: int = 3
walkforward_train_ratio: float = 0.6
```

Add to `from_dict` flattening under `evaluation` section.

**Step 2: Add timerange splitter for walk-forward**

In `_helpers.py`, add:
```python
def _walkforward_timeranges(
    timerange: str, *, folds: int = 3, train_ratio: float = 0.6
) -> list[tuple[str, str]]:
    """Split YYYYMMDD-YYYYMMDD into expanding-window walk-forward folds.

    Returns list of (train_range, test_range) tuples.
    """
```

**Step 3: Write test**

```python
def test_walkforward_timeranges_basic():
    from agent_market.strategy_miner._helpers import _walkforward_timeranges
    folds = _walkforward_timeranges("20250101-20250401", folds=3, train_ratio=0.6)
    assert len(folds) == 3
    for train_range, test_range in folds:
        assert "-" in train_range and "-" in test_range
```

Run: `pytest tests/test_walkforward.py -v`

---

## Task 8: Walk-forward — Wire into phase_evaluation

**Files:**
- Modify: `src/agent_market/strategy_miner/_evaluation.py` (or phases.py)
- Create: `tests/test_walkforward.py`

**Step 1: Add walk-forward backtest loop in phase_evaluation**

When `config.walkforward_enabled`:
1. Split timerange into N folds via `_walkforward_timeranges`
2. For each fold, run freqtrade backtest on test_range
3. Collect per-fold Sharpe/profit
4. Final score = mean(fold_scores); add `walkforward_std` penalty if too high
5. Store fold results in candidate.backtest_summary["walkforward"]

**Step 2: Write integration test**

Test with mock subprocess that returns fake backtest ZIPs.

**Step 3: Run tests**

---

## Task 9: Complete dependency manifest

**Files:**
- Modify: `requirements-full.txt`
- Modify: `constraints.txt`

**Step 1: Add missing ML/RL dependencies to requirements-full.txt**

```
-r requirements.txt
-r server/requirements.txt
-r requirements-dev.txt

# Golden path extras (ml + backtest)
freqtrade
lightgbm

# ML model alternatives
xgboost

# Deep learning
torch

# Reinforcement learning
stable-baselines3
gymnasium

# Portfolio optimization
PyPortfolioOpt
```

**Step 2: Ensure constraints.txt covers all**

Add missing pins:
```
stable-baselines3==2.6.*
gymnasium==1.1.*
```

**Step 3: Verify install resolves**

Run: `pip install --dry-run -c constraints.txt -r requirements-full.txt`

---

## Task 10: Broaden CI coverage

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add py_compile step to smoke job**

After "Install deps" step, add:
```yaml
      - name: Compile check (catch syntax errors + merge conflicts)
        run: |
          python -m py_compile server/app.py
          python -m py_compile server/api/routes/run.py
          python -c "
          import ast, pathlib, sys
          errors = []
          for f in pathlib.Path('src').rglob('*.py'):
              try: ast.parse(f.read_text())
              except SyntaxError as e: errors.append(f'{f}: {e}')
          for f in pathlib.Path('server').rglob('*.py'):
              try: ast.parse(f.read_text())
              except SyntaxError as e: errors.append(f'{f}: {e}')
          if errors:
              print('Syntax errors found:')
              for e in errors: print(f'  {e}')
              sys.exit(1)
          print(f'All files OK')
          "
```

**Step 2: Add import smoke step**

```yaml
      - name: Import smoke (core modules loadable)
        run: |
          python -c "from agent_market.strategy_miner import MinerConfig, MinerState, Phase"
          python -c "from agent_market.backtest_results import build_backtest_summary"
          python -c "from agent_market.strategy_miner.phases import phase_strategy_gen, phase_backtest, phase_evaluation"
```

**Step 3: Broaden pytest coverage in smoke job**

```yaml
      - name: Run pytest (extended smoke)
        run: |
          pytest -q \
            tests/test_api_smoke.py \
            tests/test_no_bom.py \
            tests/test_security_and_gates.py \
            tests/test_workspace_core.py \
            tests/test_strategy_miner_runner.py \
            tests/test_strategy_miner_artifacts.py \
            tests/test_backtest_results.py \
            tests/test_pipeline_leakage.py
```

**Step 4: Run CI locally to verify**

Run: `pytest -q tests/test_api_smoke.py tests/test_no_bom.py tests/test_security_and_gates.py tests/test_workspace_core.py tests/test_strategy_miner_runner.py tests/test_strategy_miner_artifacts.py tests/test_backtest_results.py tests/test_pipeline_leakage.py`

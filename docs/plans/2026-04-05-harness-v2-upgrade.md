# Harness V2: From Pipeline to Experiment Operating System

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the strategy mining system from a fixed 200-iteration pipeline into an adaptive experiment operating system with sealed holdout, Strategy IR, bandit scheduling, lineage tracking, and trace grading.

**Architecture:** Four-layer redesign: Control Plane (experiment DAG + bandit scheduler), Representation Plane (Strategy IR/DSL with module decoupling), Memory Plane (strategy cards + failure cards + lineage graph), Evaluation Plane (candidate eval + harness eval + promotion eval). Each layer is implemented incrementally without breaking the existing pipeline.

**Tech Stack:** Python 3.11+, Freqtrade, LightGBM/XGBoost, Optuna, pytest

---

## Phase 1: Foundation (P0 — do first, ~4-6 hours)

These 3 changes are prerequisites for everything else.

### Task 1: Sealed Final Holdout

**Problem:** The fixed OOS window (3/12-3/29) is compared 200× across iterations, making it a "selection set" not a true holdout. Leaderboard improves but live performance may not.

**Files:**
- Modify: `src/agent_market/strategy_miner/dtypes.py`
- Modify: `src/agent_market/strategy_miner/_evaluation.py`
- Create: `src/agent_market/strategy_miner/_holdout.py`
- Modify: `configs/strategy_miner_intraday_gate_expanded.json`
- Test: `tests/test_holdout.py`

**Design:**
Split the current timerange into 3 segments:
```
train:     2/28 ──── 3/11  (11 days, ML training)
selection:  3/12 ──── 3/23  (11 days, iteration scoring & champion selection)
holdout:   3/24 ──── 3/29  (5 days, sealed final validation, touched ONCE)
```

Add to MinerConfig:
```python
selection_timerange: str = ""   # used for iteration scoring (replaces current timerange)
holdout_timerange: str = ""     # sealed, used only at run completion
holdout_min_profit_pct: float = 0.0  # must be profitable on holdout
```

Add `_holdout.py` with:
```python
def run_sealed_holdout(candidate, config) -> dict:
    """Run exactly once at end of mining on the sealed holdout window."""
    # Backtest on holdout_timerange
    # Return holdout_summary with profit, trades, sharpe, etc.
    # Compare selection performance vs holdout performance
    # Flag if delta > threshold (overfitting signal)
```

Wire into `runner.py`: after mining completes (Phase.COMPLETE), run sealed holdout on best_candidate.

**Step 1:** Add config fields to dtypes.py
**Step 2:** Create `_holdout.py` with `run_sealed_holdout()`
**Step 3:** Wire into runner.py after mining loop
**Step 4:** Update configs: selection=3/12-3/23, holdout=3/24-3/29
**Step 5:** Add test
**Step 6:** Commit

---

### Task 2: Reorder Phases — Cheap Checks Before Expensive Search

**Problem:** Currently hyperopt (60 epochs × 16 jobs) runs BEFORE static validation catches syntax errors. Expensive work wasted on broken candidates.

**Files:**
- Modify: `src/agent_market/strategy_miner/_backtest.py`

**Design:**
Reorder phase_backtest to:
1. Static validation (free, ~0s)
2. Freqtrade list-strategies preflight (cheap, ~5s)
3. 2-pair smoke backtest (cheap, ~10s) — NEW
4. Hyperopt (expensive, ~60s) — only if smoke passes
5. Full backtest (expensive, ~30s) — only if hyperopt didn't fail

**Step 1:** Extract smoke backtest logic into `_run_smoke_backtest()`
```python
def _run_smoke_backtest(candidate, config, sandbox, strategies_dir) -> tuple[bool, dict]:
    """Quick 2-pair backtest to reject obvious failures before hyperopt."""
    quick_pairs = ["BTC/USDT", "ETH/USDT"]
    # Run backtest on just 2 pairs, short timeout
    # Pass if trades > 0 and no crash
    # Return (passed, summary)
```

**Step 2:** Move hyperopt call AFTER smoke backtest passes
**Step 3:** Add logging: "Smoke passed, proceeding to hyperopt" / "Smoke failed, skipping hyperopt"
**Step 4:** Test
**Step 5:** Commit

---

### Task 3: Family Bandit Budget Allocator

**Problem:** Fixed 5 slots (3 rule + 2 ML) wastes budget on families that consistently fail. Need adaptive allocation.

**Files:**
- Create: `src/agent_market/strategy_miner/_scheduler.py`
- Modify: `src/agent_market/strategy_miner/_generation.py`
- Modify: `src/agent_market/strategy_miner/dtypes.py`
- Test: `tests/test_scheduler.py`

**Design:**
Replace fixed slot rotation with Thompson Sampling bandit:

```python
@dataclass
class FamilyStats:
    family: str          # "rule/mean-reversion", "ml/lightgbm", etc.
    trials: int = 0
    successes: int = 0   # passed constraints
    total_reward: float = 0.0
    avg_reward: float = 0.0

class BanditScheduler:
    def __init__(self, families: list[str], exploration_bonus: float = 1.0):
        self.stats: dict[str, FamilyStats] = {}

    def select_families(self, n: int) -> list[str]:
        """Thompson Sampling: sample from Beta(successes+1, failures+1) for each family."""
        # At least 1 slot for exploration (random family)
        # Remaining slots allocated by Thompson Sampling score

    def update(self, family: str, reward: float, passed: bool):
        """Update stats after candidate evaluation."""
```

Wire into `phase_strategy_gen`: replace fixed `_INDICATOR_SETS` / `_candidate_type_for_slot` with `scheduler.select_families(n=5)`.

Persist scheduler state in checkpoint.

**Step 1:** Create `_scheduler.py` with `BanditScheduler`
**Step 2:** Add `FamilyStats` serialization to checkpoint
**Step 3:** Wire into `_generation.py`
**Step 4:** Add test for Thompson Sampling selection
**Step 5:** Commit

---

## Phase 2: Representation & Memory (P1 — ~8-12 hours)

### Task 4: Strategy IR / DSL

**Problem:** LLM writes complete strategy code, mixing alpha/exit/execution/risk. Can't reuse modules across candidates.

**Files:**
- Create: `src/agent_market/strategy_miner/ir.py`
- Create: `src/agent_market/strategy_miner/compiler.py`
- Test: `tests/test_strategy_ir.py`

**Design:**
```python
@dataclass
class StrategyIR:
    """Intermediate representation separating strategy concerns."""
    name: str
    timeframe: str

    # Alpha module: what signals to use
    alpha: AlphaModule  # indicators, entry conditions, exit conditions

    # Regime filter: when to trade
    regime: RegimeModule  # ADX/volatility filter, trend detection

    # Exit module: how to exit
    exit: ExitModule  # ROI, stoploss, trailing, time-based

    # Execution module: position management
    execution: ExecutionModule  # DCA, grid, martingale, leverage

    # Risk overlay: portfolio-level controls
    risk: RiskModule  # max drawdown, position sizing, pair limits

    # Search space: what Hyperopt can tune
    search_space: dict[str, ParameterSpec]

class StrategyCompiler:
    def compile(self, ir: StrategyIR) -> str:
        """Compile IR into Freqtrade IStrategy Python code."""
```

**Step 1:** Define IR dataclasses
**Step 2:** Implement compiler (IR → Python code)
**Step 3:** Update prompts to generate IR JSON instead of raw code
**Step 4:** Test: IR → compile → py_compile → backtest
**Step 5:** Commit

---

### Task 5: Knowledge Base Upgrade — Strategy Cards + Lineage Graph

**Problem:** Current KB is flat archive. No lineage tracking, no module-level reuse.

**Files:**
- Create: `src/agent_market/strategy_miner/memory.py`
- Modify: `src/agent_market/strategy_miner/knowledge_base.py`
- Test: `tests/test_memory.py`

**Design:**
```python
@dataclass
class StrategyCard:
    """Rich card for a strategy with provenance and performance."""
    id: str
    name: str
    parent_ids: list[str]          # lineage
    mutation_description: str       # what changed from parent
    mutation_axis: str             # which module was changed
    ir: Optional[StrategyIR]       # structured representation

    # Performance
    selection_metrics: dict        # on selection set
    holdout_metrics: Optional[dict] # on sealed holdout
    per_pair_metrics: dict
    regime_performance: dict       # performance by market regime

    # Status
    status: str  # "exploring", "promising", "champion", "retired", "failed"
    failure_taxonomy: list[str]    # structured failure reasons

@dataclass
class LineageGraph:
    """Track parent-child relationships between candidates."""
    nodes: dict[str, StrategyCard]
    edges: list[tuple[str, str, str]]  # parent_id, child_id, mutation_type

    def ancestors(self, card_id: str) -> list[StrategyCard]: ...
    def best_in_family(self, family: str) -> Optional[StrategyCard]: ...
    def similar_failures(self, card: StrategyCard) -> list[StrategyCard]: ...
```

**Step 1:** Define StrategyCard and LineageGraph
**Step 2:** Implement serialization (JSON)
**Step 3:** Wire into evaluation: create card after each candidate
**Step 4:** Update retrieval: use lineage for context instead of flat top-3
**Step 5:** Test
**Step 6:** Commit

---

### Task 6: Candidate State Machine + Progress Tracking

**Problem:** No structured progress per candidate. Hard to debug why specific candidates fail.

**Files:**
- Create: `src/agent_market/strategy_miner/_candidate_sm.py`
- Modify: `src/agent_market/strategy_miner/dtypes.py`

**Design:**
```python
class CandidateStage(Enum):
    IDEA = "idea"
    SPEC = "spec"
    RENDERED = "rendered"
    STATIC_PASS = "static_pass"
    SMOKE_PASS = "smoke_pass"
    HYPEROPT_DONE = "hyperopt_done"
    BACKTEST_DONE = "backtest_done"
    EVALUATED = "evaluated"
    HOLDOUT_TESTED = "holdout_tested"
    PROMOTED = "promoted"
    FAILED = "failed"

# Add to StrategyCandidate:
stage: CandidateStage = CandidateStage.IDEA
stage_history: list[dict] = []  # timestamp, stage, duration, result
```

**Step 1:** Add CandidateStage enum
**Step 2:** Wire stage transitions into each phase
**Step 3:** Log stage transitions with timing
**Step 4:** Commit

---

## Phase 3: Evaluation & Production (P2 — ~6-8 hours)

### Task 7: Trace Grading

**Files:**
- Create: `src/agent_market/strategy_miner/_trace_grader.py`

**Design:**
```python
def grade_trace(candidate: StrategyCandidate) -> dict:
    """Grade the agent trace for quality issues."""
    return {
        "retrieval_relevant": bool,    # was context useful?
        "hypothesis_clear": bool,      # was the idea well-formed?
        "patch_targeted": bool,        # did repair target the actual failure?
        "premature_stop": bool,        # did it give up too early?
        "wasted_cycles": int,          # how many repair attempts were useless?
        "tool_misuse": list[str],      # incorrect tool usage
        "overall_grade": float,        # 0-1
    }
```

### Task 8: Retrieval Restructure

Replace flat "last 5 + top 3" with JIT retrieval from lineage graph:
- 2 most similar successful cards
- 2 most similar failure cards
- 1 counter-example (worked in different regime)

### Task 9: Shadow Live Interface

Add `phase_shadow` after sealed holdout:
- Paper trading simulation
- Champion-challenger comparison
- Drift detection hooks

### Task 10: Sync to Remote Server

After all local changes pass tests:
```bash
git push origin main
ssh zechuan@222.200.185.120 "cd Agent_market && git pull && ..."
```

---

## Execution Order

```
Phase 1 (P0): Tasks 1-3 — Foundation
  → Test locally → Push → Sync to 120 → Restart miners

Phase 2 (P1): Tasks 4-6 — Representation & Memory
  → Test locally → Push → Sync → Restart

Phase 3 (P2): Tasks 7-10 — Evaluation & Production
  → Test locally → Push → Sync → Restart
```

## Success Metrics (from user's requirements)

1. **Sealed-holdout delta**: selection vs holdout performance gap < 50%
2. **Best score / GPU-hour**: improvement per compute unit
3. **Exploration yield**: >10% of candidates reach full eval (currently 2%)
4. **Skill reuse rate**: >30% of new candidates use existing module
5. **Trace pass rate**: >70% of traces rated "good" by grader

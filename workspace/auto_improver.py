"""Auto-Improver — LLM-driven autonomous strategy optimizer.

The agent reads experiment history, analyzes failures with LLM,
generates improved strategy code, backtests, evaluates, and iterates.

Usage:
    from workspace.auto_improver import AutoImprover
    improver = AutoImprover()
    report = improver.run_cycle(max_iterations=5)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workspace.backtest_api import run_backtest
from workspace.evaluator import evaluate
from workspace.tracker import record_experiment, list_experiments, query_best


class AutoImprover:
    """LLM-driven autonomous strategy optimizer."""

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL", "http://localhost:4141/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "_")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.2")
        self.timeout = timeout
        self.max_retries = max_retries
        self.strategies_dir = ROOT / "workspace" / "strategies"
        self.results_dir = ROOT / "workspace" / "results"

    # ------------------------------------------------------------------
    # LLM interface
    # ------------------------------------------------------------------

    def _llm_call(self, system: str, user: str, *, temperature: float = 0.4) -> str:
        """Call LLM via opencode run (uses project .opencode.json config).

        Falls back to direct HTTP if opencode is not available.
        """
        prompt = f"{system}\n\n{user}" if system else user

        for attempt in range(self.max_retries):
            try:
                # Primary: opencode run
                proc = subprocess.run(
                    [
                        "opencode", "run",
                        "-m", f"custom/{self.model}",
                        "--format", "json",
                        prompt,
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if proc.returncode == 0:
                    # Parse JSON event stream — extract assistant text
                    return self._parse_opencode_output(proc.stdout)
                # Fallback: direct HTTP
                return self._llm_call_http(system, user, temperature=temperature)
            except subprocess.TimeoutExpired:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"opencode run timed out after {self.timeout}s")
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {exc}")
        return ""

    def _parse_opencode_output(self, raw: str) -> str:
        """Extract assistant text from opencode --format json NDJSON stream.

        Event format: {"type":"text","part":{"type":"text","text":"..."}}
        """
        text_parts = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    continue
                # opencode v1.3: type="text", part.text=content
                if event.get("type") == "text":
                    part = event.get("part", {})
                    text = part.get("text", "")
                    if text:
                        text_parts.append(str(text))
            except json.JSONDecodeError:
                continue
        return "\n".join(text_parts) if text_parts else raw

    def _llm_call_http(self, system: str, user: str, *, temperature: float = 0.4) -> str:
        """Fallback: direct HTTP call to OpenAI-compatible API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4096,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Step 1: Analyze history
    # ------------------------------------------------------------------

    def analyze_history(self) -> str:
        """Read experiment history and produce LLM analysis of failures."""
        experiments = list_experiments()
        best = query_best("sharpe", 5)

        history_text = "EXPERIMENT HISTORY (sorted by sharpe, best first):\n"
        for e in best:
            m = e["metrics"]
            history_text += (
                f"  #{e['id']} {e['strategy_name']}: "
                f"Sharpe={m['sharpe']:.2f}, Profit={m['profit_pct']:+.2f}%, "
                f"DD={m['max_drawdown_pct']:.2f}%, WR={m['win_rate']:.0%}, "
                f"PF={m['profit_factor']:.2f}, Trades={m['trades']}\n"
            )
            if e.get("notes"):
                history_text += f"    Notes: {e['notes']}\n"

        # Include best strategy source code if available
        best_code = ""
        if best:
            best_path = ROOT / best[0].get("strategy_path", "")
            if best_path.exists():
                best_code = best_path.read_text(encoding="utf-8")

        system = textwrap.dedent("""
            You are a senior quantitative researcher analyzing trading strategy backtest results.
            The market is BTC/ETH on KuCoin, 1H timeframe, ~60 days, overall market -2.47% (choppy/bearish).
            Available data columns: date, open, high, low, close, volume.
            All strategies are long-only.

            Analyze the experiment history and identify:
            1. Why each strategy failed (specific technical reasons)
            2. What patterns work (high win rate strategies) vs what doesn't
            3. A concrete hypothesis for a NEW strategy that could achieve Sharpe > 0
            4. Specific indicators and parameters to use

            Be precise and quantitative, not generic.
        """).strip()

        user = f"{history_text}\n\nBest strategy code:\n```python\n{best_code}\n```\n\nAnalyze and propose a new strategy."

        analysis = self._llm_call(system, user, temperature=0.3)
        return analysis

    # ------------------------------------------------------------------
    # Step 2: Generate strategy
    # ------------------------------------------------------------------

    def generate_strategy(self, analysis: str, iteration: int) -> Path:
        """Have LLM write a new strategy based on analysis."""
        strategy_name = f"AutoStrategy_v{iteration}"
        file_name = f"auto_v{iteration}.py"
        out_path = self.strategies_dir / file_name

        # Read existing best strategy as reference
        best = query_best("sharpe", 1)
        reference_code = ""
        if best:
            ref_path = ROOT / best[0].get("strategy_path", "")
            if ref_path.exists():
                reference_code = ref_path.read_text(encoding="utf-8")

        system = textwrap.dedent(f"""
            You are an expert Python developer writing freqtrade trading strategies.
            Output ONLY valid Python code, no markdown, no explanation.

            DATA STATISTICS (BTC/USDT 1H, ~60 days):
            - close: 84488-97656, mean=90276, std=2526
            - hourly_return: mean~0%, std=0.43%, range [-3.7%, +2.6%]
            - RSI(14): range 8-91, mean=50
            - BB_width(20): range 0.001-0.056, mean=0.013
            - volume_ratio (vs 20-bar avg): range 0.08-6.4, mean=1.04
            - Market is CHOPPY/BEARISH (-2.47% total). Trend strategies fail.

            CRITICAL: entry conditions must NOT be too strict! With ~1448 bars,
            your strategy should generate 20-100 trades. If conditions are too
            narrow you get 0 trades. Use relaxed thresholds.

            Requirements:
            - Class name must be: {strategy_name}
            - Must inherit from freqtrade.strategy.IStrategy
            - Must implement: populate_indicators, populate_entry_trend, populate_exit_trend
            - timeframe = "1h"
            - can_short = False (long only)
            - Use numpy and pandas only (no ta-lib, no external indicators)
            - Max 150 lines
            - Include this exact import block at the top:

            from __future__ import annotations
            import sys
            from pathlib import Path
            import numpy as np
            from pandas import DataFrame
            _ROOT = Path(__file__).resolve().parents[2]
            if str(_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(_ROOT / "src"))
                sys.path.insert(0, str(_ROOT))
            from freqtrade.strategy import IStrategy
        """).strip()

        user = f"Analysis of previous strategies:\n{analysis}\n\n"
        if reference_code:
            user += f"Current best strategy for reference:\n```python\n{reference_code}\n```\n\n"
        user += f"Write a NEW and IMPROVED strategy class named {strategy_name}. Focus on the weaknesses identified in the analysis."

        code = self._llm_call(system, user, temperature=0.4)

        # Extract code from markdown if wrapped
        code = self._extract_code(code)

        out_path.write_text(code, encoding="utf-8")
        return out_path

    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response (may be wrapped in markdown)."""
        # Try to find ```python ... ``` block
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        # If no markdown, assume raw code
        return text.strip()

    # ------------------------------------------------------------------
    # Step 3: Validate strategy
    # ------------------------------------------------------------------

    def validate_strategy(self, path: Path) -> tuple[bool, str]:
        """Validate strategy syntax and structure. Returns (ok, error_message)."""
        try:
            code = path.read_text(encoding="utf-8")
        except Exception as exc:
            return False, f"Cannot read file: {exc}"

        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        # Check for IStrategy subclass
        has_strategy_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name == "IStrategy":
                        has_strategy_class = True

        if not has_strategy_class:
            return False, "No IStrategy subclass found"

        # Check required methods
        required_methods = {"populate_indicators", "populate_entry_trend", "populate_exit_trend"}
        found_methods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in required_methods:
                found_methods.add(node.name)

        missing = required_methods - found_methods
        if missing:
            return False, f"Missing methods: {', '.join(sorted(missing))}"

        return True, "OK"

    def fix_strategy(self, path: Path, error: str) -> bool:
        """Have LLM fix a broken strategy. Returns True if fixed."""
        code = path.read_text(encoding="utf-8")

        system = textwrap.dedent("""
            You are fixing a Python freqtrade strategy that has errors.
            Output ONLY the complete fixed Python code, no markdown, no explanation.
            Keep the same class name and logic, just fix the errors.
        """).strip()

        user = f"This strategy has an error:\n{error}\n\nBroken code:\n```python\n{code}\n```\n\nFix it."

        fixed_code = self._llm_call(system, user, temperature=0.1)
        fixed_code = self._extract_code(fixed_code)
        path.write_text(fixed_code, encoding="utf-8")

        ok, new_error = self.validate_strategy(path)
        return ok

    # ------------------------------------------------------------------
    # Step 4: Run and evaluate
    # ------------------------------------------------------------------

    def run_and_evaluate(self, path: Path, iteration: int) -> Dict[str, Any]:
        """Backtest + evaluate + record. Returns combined result."""
        from workspace.backtest_api import _detect_strategy_name

        strategy_name = _detect_strategy_name(path) or f"AutoStrategy_v{iteration}"

        bt_result = run_backtest(str(path), strategy_name, timerange="20251126-20260125")
        evaluation = evaluate(bt_result)

        if bt_result.get("ok"):
            record_experiment(
                backtest_result=bt_result,
                evaluation=evaluation,
                strategy_name=strategy_name,
                notes=f"auto_improver iteration {iteration}",
                tags=[f"auto_v{iteration}", "auto_improver"],
            )

        return {
            "iteration": iteration,
            "strategy_name": strategy_name,
            "strategy_path": str(path),
            "backtest": bt_result,
            "evaluation": evaluation,
        }

    # ------------------------------------------------------------------
    # Step 5: Full cycle
    # ------------------------------------------------------------------

    def run_cycle(self, max_iterations: int = 5) -> Dict[str, Any]:
        """Run the full analyze → generate → validate → backtest → evaluate loop."""
        cycle_results: List[Dict[str, Any]] = []

        for i in range(1, max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"AUTO-IMPROVER ITERATION {i}/{max_iterations}")
            print(f"{'='*60}")

            # 1. Analyze
            print("[1/5] Analyzing history...")
            try:
                analysis = self.analyze_history()
                print(f"  Analysis: {analysis[:200]}...")
            except Exception as exc:
                print(f"  Analysis failed: {exc}")
                analysis = "No analysis available. Generate a simple RSI-based strategy."

            # 2. Generate
            print(f"[2/5] Generating strategy auto_v{i}...")
            try:
                path = self.generate_strategy(analysis, i)
                print(f"  Written: {path}")
            except Exception as exc:
                print(f"  Generation failed: {exc}")
                continue

            # 3. Validate (with auto-fix)
            print("[3/5] Validating...")
            ok, error = self.validate_strategy(path)
            if not ok:
                print(f"  Validation failed: {error}")
                print("  Attempting auto-fix...")
                for fix_attempt in range(self.max_retries):
                    fixed = self.fix_strategy(path, error)
                    if fixed:
                        print(f"  Fixed on attempt {fix_attempt + 1}")
                        ok = True
                        break
                    ok, error = self.validate_strategy(path)
                    print(f"  Fix attempt {fix_attempt + 1} failed: {error}")
            if not ok:
                print(f"  SKIP: could not fix strategy")
                cycle_results.append({"iteration": i, "error": error})
                continue
            print("  Validation OK")

            # 4. Run + Evaluate
            print("[4/5] Running backtest + evaluation...")
            try:
                result = self.run_and_evaluate(path, i)
                bt = result["backtest"]
                ev = result["evaluation"]
                if bt.get("ok"):
                    print(f"  Trades={bt['trades']}, Sharpe={bt['sharpe']:.4f}, "
                          f"Profit={bt['profit_pct']}%, Score={ev['total_score']}/100 ({ev['grade']})")
                else:
                    print(f"  Backtest failed: {bt.get('error', '?')[:200]}")
                    # Try to fix runtime errors
                    if "Error" in str(bt.get("error", "")):
                        print("  Attempting runtime fix...")
                        fixed = self.fix_strategy(path, str(bt.get("error", "")))
                        if fixed:
                            result = self.run_and_evaluate(path, i)
                            bt = result["backtest"]
                            ev = result["evaluation"]
                            if bt.get("ok"):
                                print(f"  Fixed! Trades={bt['trades']}, Sharpe={bt['sharpe']:.4f}")
            except Exception as exc:
                print(f"  Run failed: {exc}")
                result = {"iteration": i, "error": str(exc)}

            cycle_results.append(result)

            # 5. Summary
            print(f"[5/5] Iteration {i} complete")

        # Final report
        best_after = query_best("sharpe", 1)
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations_run": len(cycle_results),
            "successful": sum(1 for r in cycle_results if r.get("backtest", {}).get("ok")),
            "results": [
                {
                    "iteration": r.get("iteration"),
                    "strategy": r.get("strategy_name", "?"),
                    "sharpe": r.get("backtest", {}).get("sharpe"),
                    "profit_pct": r.get("backtest", {}).get("profit_pct"),
                    "score": r.get("evaluation", {}).get("total_score"),
                    "error": r.get("error"),
                }
                for r in cycle_results
            ],
            "best_overall": {
                "strategy": best_after[0]["strategy_name"] if best_after else None,
                "sharpe": best_after[0]["metrics"]["sharpe"] if best_after else None,
            } if best_after else None,
        }

        report_path = self.results_dir / f"auto_improver_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport: {report_path}")

        return report

    # ------------------------------------------------------------------
    # Parameter sweep for best strategy
    # ------------------------------------------------------------------

    def parameter_sweep(self, strategy_path: str | Path, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Grid search over strategy parameters by rewriting code."""
        path = Path(strategy_path)
        original_code = path.read_text(encoding="utf-8")
        results = []

        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            # Rewrite parameter values in strategy code
            code = original_code
            for param_name, param_value in params.items():
                # Match: param_name = <value>
                pattern = rf"({param_name}\s*=\s*)[^\n]+"
                replacement = rf"\g<1>{param_value}"
                code = re.sub(pattern, replacement, code, count=1)

            path.write_text(code, encoding="utf-8")
            bt = run_backtest(str(path), timerange="20251126-20260125")
            ev = evaluate(bt)

            result = {
                "params": params,
                "ok": bt.get("ok", False),
                "sharpe": bt.get("sharpe", 0),
                "profit_pct": bt.get("profit_pct", 0),
                "score": ev.get("total_score", 0),
            }
            results.append(result)

            if bt.get("ok"):
                record_experiment(
                    backtest_result=bt,
                    evaluation=ev,
                    strategy_name=f"sweep_{path.stem}",
                    notes=f"param_sweep: {params}",
                    tags=["param_sweep"],
                )

        # Restore original
        path.write_text(original_code, encoding="utf-8")
        results.sort(key=lambda r: r.get("sharpe", -999), reverse=True)
        return results


__all__ = ["AutoImprover"]

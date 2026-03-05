"""Strategy-level evolution: mutation and crossover operators."""
from __future__ import annotations

import ast
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter mutation helpers
# ---------------------------------------------------------------------------

_NUMERIC_PARAM_RE = re.compile(
    r"(?P<name>minimal_roi|stoploss|timeframe|trailing_stop_positive|"
    r"trailing_stop_positive_offset|roi_t\d+|buy_\w+|sell_\w+)\s*=\s*"
    r"(?P<value>-?[\d.]+)"
)

_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]


def _perturb_numeric(value: float, scale: float = 0.2) -> float:
    """Perturb a numeric value by ±scale fraction."""
    delta = value * scale * (random.random() * 2 - 1)
    return round(value + delta, 6)


def mutate_parameters(code: str, intensity: float = 0.3) -> str:
    """Randomly perturb numeric strategy parameters.

    Args:
        code: Strategy source code.
        intensity: Fraction of parameters to mutate (0-1).

    Returns:
        Mutated code.
    """
    matches = list(_NUMERIC_PARAM_RE.finditer(code))
    if not matches:
        return code

    to_mutate = random.sample(
        matches, k=max(1, int(len(matches) * intensity))
    )
    # Apply replacements from right to left (descending span) to keep indices valid.
    to_mutate.sort(key=lambda m: m.start(), reverse=True)

    result = code
    for m in to_mutate:
        try:
            old_val = float(m.group("value"))
            new_val = _perturb_numeric(old_val)
            old_text = m.group(0)
            new_text = old_text.replace(m.group("value"), str(new_val))
            result = result[:m.start()] + new_text + result[m.end():]
        except (ValueError, IndexError):
            continue

    return result


# ---------------------------------------------------------------------------
# Indicator mutation
# ---------------------------------------------------------------------------

_INDICATOR_ALTERNATIVES = {
    "RSI": ["STOCHRSI", "WILLR", "CCI", "MFI"],
    "MACD": ["PPO", "TRIX", "APO"],
    "SMA": ["EMA", "WMA", "DEMA", "TEMA", "KAMA"],
    "EMA": ["SMA", "WMA", "DEMA", "TEMA", "KAMA"],
    "BBANDS": ["KELTNER", "DONCHIAN"],
    "ATR": ["NATR", "TRANGE"],
    "ADX": ["AROON", "DX", "PLUS_DI"],
    "STOCH": ["STOCHF", "STOCHRSI"],
    "OBV": ["AD", "ADOSC"],
}


def mutate_indicators(code: str, n_swaps: int = 1) -> str:
    """Swap indicator names with alternatives.

    Best-effort: swaps ta-lib style indicator names found in code.
    """
    for _ in range(n_swaps):
        for indicator, alternatives in _INDICATOR_ALTERNATIVES.items():
            # Match patterns like ta.RSI( or talib.RSI( or ta.abstract.RSI(
            pattern = re.compile(
                rf"(ta(?:lib)?(?:\.abstract)?\.)({re.escape(indicator)})\b",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(code))
            if matches:
                m = random.choice(matches)
                replacement = random.choice(alternatives)
                code = code[:m.start(2)] + replacement + code[m.end(2):]
                break
    return code


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def _extract_method_body(code: str, method_name: str) -> Optional[str]:
    """Extract the body of a method from strategy code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == method_name:
                        return ast.get_source_segment(code, item)
    return None


def crossover_entry_exit(parent_a: str, parent_b: str) -> str:
    """Cross two strategies: take entry from A, exit from B.

    Falls back to parent_a if extraction fails.
    """
    exit_b = _extract_method_body(parent_b, "populate_exit_trend")
    if exit_b is None:
        return parent_a

    exit_a = _extract_method_body(parent_a, "populate_exit_trend")
    if exit_a is None:
        return parent_a

    return parent_a.replace(exit_a, exit_b)


# ---------------------------------------------------------------------------
# Ensemble: build combined indicator from multiple strategies
# ---------------------------------------------------------------------------

def build_ensemble_indicators(strategies: List[Dict[str, Any]]) -> str:
    """Generate a code snippet that combines signals from multiple strategies.

    Each strategy contributes a binary signal; the ensemble uses majority vote.
    """
    lines = [
        "    # === Ensemble signals from elite strategies ===",
        "    ensemble_score = 0",
    ]
    for i, s in enumerate(strategies[:5]):
        name = s.get("name", f"strat_{i}")
        lines.append(f"    # Signal from {name} (reward={s.get('reward', '?'):.4f})")
        lines.append(f"    ensemble_score += dataframe['enter_long'].astype(int)  # placeholder_{i}")
    lines.append(f"    dataframe['ensemble_vote'] = ensemble_score >= {max(1, len(strategies[:5]) // 2 + 1)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level evolve function
# ---------------------------------------------------------------------------

def evolve_strategy(
    code: str,
    *,
    elite_codes: Optional[List[str]] = None,
    mutation_intensity: float = 0.3,
    indicator_swaps: int = 1,
    crossover_prob: float = 0.3,
) -> Tuple[str, List[str]]:
    """Evolve a strategy through mutation and optional crossover.

    Returns (evolved_code, list_of_applied_operations).
    """
    ops: List[str] = []

    # Parameter mutation
    mutated = mutate_parameters(code, intensity=mutation_intensity)
    if mutated != code:
        ops.append("param_mutation")
        code = mutated

    # Indicator swap
    swapped = mutate_indicators(code, n_swaps=indicator_swaps)
    if swapped != code:
        ops.append("indicator_swap")
        code = swapped

    # Crossover with an elite
    if elite_codes and random.random() < crossover_prob:
        donor = random.choice(elite_codes)
        crossed = crossover_entry_exit(code, donor)
        if crossed != code:
            ops.append("crossover_exit")
            code = crossed

    if not ops:
        ops.append("no_change")

    return code, ops

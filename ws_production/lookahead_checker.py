"""Lookahead Bias Checker — detects future data leakage in strategy/model code.

Usage:
    from workspace.lookahead_checker import check_lookahead, LookaheadReport
    report = check_lookahead("workspace/strategies/my_strategy.py")
    if not report.ok:
        print(report.issues)
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class LookaheadIssue:
    """A single lookahead bias detection."""
    severity: str  # "critical", "warning", "info"
    line: int
    code: str
    description: str
    fix_suggestion: str


@dataclass
class LookaheadReport:
    """Result of lookahead analysis."""
    path: str
    issues: List[LookaheadIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "critical" for i in self.issues)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> str:
        if self.ok and not self.issues:
            return "No lookahead issues detected."
        parts = []
        for issue in self.issues:
            parts.append(f"  [{issue.severity.upper()}] L{issue.line}: {issue.description}")
            parts.append(f"    Code: {issue.code.strip()}")
            parts.append(f"    Fix: {issue.fix_suggestion}")
        return "\n".join(parts)


# Patterns that indicate potential lookahead bias
_CRITICAL_PATTERNS = [
    # bfill() / backfill() — fills NaN with FUTURE values
    (r'\.bfill\s*\(', "bfill() uses future values to fill NaN — causes lookahead bias",
     "Replace .bfill() with .ffill().fillna(0.0)"),
    (r'\.backfill\s*\(', "backfill() uses future values — same as bfill()",
     "Replace .backfill() with .ffill().fillna(0.0)"),

    # shift with negative values
    (r'\.shift\s*\(\s*-\d', "shift(-N) looks into the future",
     "Use shift(N) with positive N only"),

    # Direct future_return / target / label in strategy (not training)
    (r'future_return\s*\(', "future_return() computes future values — must NOT be used in strategy",
     "Remove future_return(); only use in training label generation"),

    # iloc[-1] in rolling context can be dangerous
    (r'\.iloc\s*\[\s*-1\s*\]', "iloc[-1] in a loop/apply context may cause lookahead",
     "Verify this doesn't access future data in vectorized operations"),
]

_WARNING_PATTERNS = [
    # fillna with interpolate
    (r'\.interpolate\s*\(', "interpolate() may use future values depending on method",
     "Use method='pad' or replace with .ffill()"),

    # rolling with center=True
    (r'center\s*=\s*True', "center=True in rolling window uses future values",
     "Remove center=True or set center=False"),

    # resample or groupby without explicit sort
    (r'\.resample\s*\(', "resample() may reorder data — verify sort order is preserved",
     "Ensure data is sorted by time before and after resample"),

    # Whole-dataframe operations that might leak
    (r'\.rank\s*\(\s*\)', "rank() across full dataframe includes future rows",
     "Use rolling rank or ensure rank is computed per-bar only"),

    # StandardScaler / fit_transform on full dataset
    (r'fit_transform\s*\(', "fit_transform on full dataset leaks future statistics",
     "Use expanding/rolling statistics, or fit on train only"),

    # np.mean / np.std on full array in indicators
    (r'np\.mean\s*\(\s*df\b|np\.std\s*\(\s*df\b', "np.mean/std on full dataframe includes future bars",
     "Use rolling().mean() / rolling().std() instead"),
]

_INFO_PATTERNS = [
    # bfill after ffill is ok but worth noting
    (r'\.ffill\(\)\.bfill\(\)', "ffill().bfill() — bfill only affects leading NaNs (usually acceptable)",
     "Consider .ffill().fillna(0.0) to be fully safe"),
]


def check_lookahead(path: str | Path) -> LookaheadReport:
    """Check a strategy or model .py file for lookahead bias."""
    path = Path(path)
    report = LookaheadReport(path=str(path))

    if not path.exists():
        report.issues.append(LookaheadIssue(
            severity="critical", line=0, code="",
            description=f"File not found: {path}",
            fix_suggestion="Check the file path",
        ))
        return report

    code = path.read_text(encoding="utf-8")
    lines = code.split("\n")

    # Check each pattern against each line
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern, desc, fix in _CRITICAL_PATTERNS:
            if re.search(pattern, line):
                # Special case: ffill().bfill() is actually ok
                if "bfill" in pattern and ".ffill()" in line and ".bfill()" in line:
                    report.issues.append(LookaheadIssue(
                        severity="info", line=line_num, code=line,
                        description="ffill().bfill() — bfill only fills leading NaNs",
                        fix_suggestion="Usually OK, but .ffill().fillna(0.0) is safer",
                    ))
                    continue
                report.issues.append(LookaheadIssue(
                    severity="critical", line=line_num, code=line,
                    description=desc, fix_suggestion=fix,
                ))

        for pattern, desc, fix in _WARNING_PATTERNS:
            if re.search(pattern, line):
                report.issues.append(LookaheadIssue(
                    severity="warning", line=line_num, code=line,
                    description=desc, fix_suggestion=fix,
                ))

        for pattern, desc, fix in _INFO_PATTERNS:
            if re.search(pattern, line):
                # Don't duplicate if already added
                if not any(i.line == line_num and i.severity == "info" for i in report.issues):
                    report.issues.append(LookaheadIssue(
                        severity="info", line=line_num, code=line,
                        description=desc, fix_suggestion=fix,
                    ))

    # AST-level checks
    try:
        tree = ast.parse(code)
        _check_ast_lookahead(tree, lines, report)
    except SyntaxError:
        pass

    return report


def _check_ast_lookahead(tree: ast.AST, lines: list, report: LookaheadReport) -> None:
    """AST-based lookahead checks."""
    for node in ast.walk(tree):
        # Check for negative shift in function calls
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name == "shift" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    if isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, (int, float)):
                        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        report.issues.append(LookaheadIssue(
                            severity="critical",
                            line=node.lineno,
                            code=line,
                            description=f"shift(-{arg.operand.value}) explicitly looks into the future",
                            fix_suggestion=f"Use shift({arg.operand.value}) to look at past data",
                        ))


def fix_lookahead_issues(path: str | Path) -> tuple[bool, str]:
    """Auto-fix critical lookahead issues in a file.

    Returns (changed, description).
    """
    path = Path(path)
    code = path.read_text(encoding="utf-8")
    original = code
    fixes = []

    # Fix 1: Replace standalone .bfill() with .fillna(0.0)
    if re.search(r'(?<!\.)\.bfill\s*\(\s*\)', code):
        # But NOT .ffill().bfill() which is acceptable
        # Replace .bfill() that's NOT preceded by .ffill()
        code = re.sub(r'(?<!ffill\(\))\.bfill\s*\(\s*\)', '.fillna(0.0)', code)
        fixes.append("replaced .bfill() with .fillna(0.0)")

    # Fix 2: Replace .backfill() with .fillna(0.0)
    if '.backfill()' in code:
        code = code.replace('.backfill()', '.fillna(0.0)')
        fixes.append("replaced .backfill() with .fillna(0.0)")

    # Fix 3: Replace center=True with center=False
    if 'center=True' in code:
        code = code.replace('center=True', 'center=False')
        fixes.append("replaced center=True with center=False")

    if code != original:
        path.write_text(code, encoding="utf-8")
        return True, "; ".join(fixes)
    return False, "no fixable issues found"


def check_train_test_overlap(
    train_config: dict,
    backtest_timerange: str,
    data_dir: str = "user_data/data",
) -> LookaheadIssue | None:
    """Check if training data overlaps with backtest period.

    Returns a critical issue if overlap detected, None otherwise.
    """
    try:
        import pandas as pd

        exchange = train_config.get("exchange", "kucoin")
        pairs = train_config.get("pairs", ["BTC/USDT"])
        timeframe = train_config.get("timeframe", "1h")

        # Parse backtest timerange
        parts = backtest_timerange.split("-")
        if len(parts) != 2:
            return None
        bt_start = pd.Timestamp(parts[0])
        bt_end = pd.Timestamp(parts[1])

        # Load data to see actual range
        data_path = Path(data_dir) / exchange
        for pair in pairs:
            slug = pair.replace("/", "_")
            feather = data_path / f"{slug}-{timeframe}.feather"
            if not feather.exists():
                continue
            df = pd.read_feather(feather)
            df["date"] = pd.to_datetime(df["date"])
            data_start = df["date"].min().tz_localize(None)
            data_end = df["date"].max().tz_localize(None)

            # If we're training on ALL data and backtesting on the same range...
            if data_start <= bt_start and data_end >= bt_end:
                return LookaheadIssue(
                    severity="critical",
                    line=0,
                    code=f"train data: {data_start} ~ {data_end}, backtest: {bt_start} ~ {bt_end}",
                    description=(
                        "Training data overlaps with backtest period! "
                        "Model is being evaluated on data it was trained on. "
                        "This inflates Sharpe/profit metrics."
                    ),
                    fix_suggestion=(
                        "Split data: train on first 70%, backtest on last 30%. "
                        "Or use walk-forward validation."
                    ),
                )
    except Exception as _exc:
        import logging; logging.getLogger(__name__).debug("Suppressed: %s", _exc)
    return None


__all__ = [
    "check_lookahead",
    "check_train_test_overlap",
    "fix_lookahead_issues",
    "LookaheadReport",
    "LookaheadIssue",
]

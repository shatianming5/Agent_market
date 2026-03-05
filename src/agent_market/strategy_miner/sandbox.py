"""Strategy code validation and sandbox preparation."""
from __future__ import annotations

import ast
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

from agent_market import paths

logger = logging.getLogger(__name__)

_FORBIDDEN_IMPORTS = frozenset(
    {
        "os",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "shutil",
        "pathlib",
    }
)

_FORBIDDEN_CALLS = frozenset(
    {
        "exec",
        "eval",
        "open",
        "__import__",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
    }
)

_REQUIRED_METHODS = frozenset(
    {
        "populate_indicators",
        "populate_entry_trend",
        "populate_exit_trend",
    }
)


def _is_negative_number(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float)
        )
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) < 0
    return False


def infer_strategy_class_name(code: str) -> str | None:
    """Best-effort infer the IStrategy subclass name from source."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "IStrategy":
                return node.name
    return None


def _extract_opencode_write_blocks(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    s = text
    lower = s.lower()
    out: list[str] = []
    pos = 0
    while True:
        i = lower.find("<write", pos)
        if i < 0:
            break
        j = lower.find(">", i)
        if j < 0:
            break
        k = lower.find("</write>", j)
        if k < 0:
            break
        body = s[j + 1 : k]
        if body.strip():
            out.append(body.strip("\n"))
        pos = k + len("</write>")
    return out


def _strip_tool_lines(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    cleaned: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue
        if s.startswith("<read") and s.endswith("/>"):
            continue
        if s.startswith("<bash") and s.endswith("/>"):
            continue
        if s.startswith("<edit") and s.endswith("/>"):
            continue
        if s.startswith("<write"):
            continue
        if s.startswith("</write"):
            continue
        if s.startswith("<final") or s.startswith("</final"):
            continue
        if s.startswith("<tool") or s.startswith("</tool"):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*$")


def auto_fix_strategy_code(code: str) -> tuple[str, list[str]]:
    """Local lightweight auto-fix for common LLM/codegen artifacts.

    This function is intentionally conservative: it targets *syntactic* blockers
    and common structural issues that are safe to fix locally.

    Returns (fixed_code, applied_fixes).
    """

    fixes: list[str] = []
    if not isinstance(code, str):
        return "", ["non_string_input"]

    out = code

    # 1) If tool tags exist, prefer `<write>...</write>` body.
    blocks = _extract_opencode_write_blocks(out)
    if blocks:
        best = max(blocks, key=len)
        out = best
        fixes.append("extract_write_block")

    # 2) Strip tool-tag lines.
    stripped = _strip_tool_lines(out)
    if stripped != out:
        out = stripped
        fixes.append("strip_tool_lines")

    # 3) Strip surrounding markdown fences.
    lines = out.splitlines()
    if lines and _FENCE_RE.match(lines[0].strip()):
        lines = lines[1:]
        fixes.append("strip_fence_header")
    if lines and _FENCE_RE.match(lines[-1].strip()):
        lines = lines[:-1]
        fixes.append("strip_fence_footer")
    out = "\n".join(lines).strip() + "\n"

    # 4) Avoid NameError from runtime-evaluated annotations (DataFrame, Order, etc.).
    if "from __future__ import annotations" not in out.splitlines()[:5]:
        if "DataFrame" in out or "Order" in out or "Trade" in out:
            out = "from __future__ import annotations\n\n" + out
            fixes.append("add_future_annotations")

    # 5) Common inheritance mismatch: class Foo(Strategy) -> class Foo(IStrategy)
    if "IStrategy" in out and "class" in out:
        # Only replace simple `(Strategy)` occurrences.
        updated = re.sub(r"\(\s*Strategy\s*\)", "(IStrategy)", out)
        if updated != out:
            out = updated
            fixes.append("fix_inheritance_strategy_to_istrategy")

    return out, fixes


def auto_fix_strategy_file(path: Path) -> tuple[bool, list[str]]:
    """Apply `auto_fix_strategy_code` to an on-disk strategy file."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False, ["read_failed"]

    fixed, fixes = auto_fix_strategy_code(raw)
    if not fixes:
        return False, []

    try:
        Path(path).write_text(fixed, encoding="utf-8")
    except Exception:
        return False, fixes + ["write_failed"]

    return True, fixes


def validate_strategy_code(code: str) -> Tuple[bool, str]:
    """AST-based static validation of strategy code.

    Returns (passed, message).
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    has_istrategy_base = False
    found_methods: set[str] = set()

    for node in ast.walk(tree):
        # Check forbidden imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in _FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in _FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {node.module}"

        # Check forbidden calls + basic anti-lookahead constraints
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in _FORBIDDEN_CALLS:
                return False, f"Forbidden call: {name}()"

            # Anti look-ahead checks (common future leakage patterns).
            if isinstance(func, ast.Attribute) and func.attr in {"shift", "pct_change", "diff"}:
                if node.args and _is_negative_number(node.args[0]):
                    return False, f"Potential look-ahead detected: {func.attr}() with negative periods"
                for kw in node.keywords or []:
                    if kw.arg in {"periods", "n"} and _is_negative_number(kw.value):
                        return False, f"Potential look-ahead detected: {func.attr}() with negative periods"

            # rolling(center=True) leaks future points.
            if isinstance(func, ast.Attribute) and func.attr == "rolling":
                for kw in node.keywords or []:
                    if (
                        kw.arg == "center"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        return False, "Potential look-ahead detected: rolling(center=True)"

            # backfill uses future values.
            if isinstance(func, ast.Attribute) and func.attr in {"bfill", "backfill"}:
                return False, f"Potential look-ahead detected: {func.attr}()"

            if isinstance(func, ast.Attribute) and func.attr == "fillna":
                for kw in node.keywords or []:
                    if kw.arg == "method" and isinstance(kw.value, ast.Constant):
                        method = str(kw.value.value).strip().lower()
                        if method in {"bfill", "backfill"}:
                            return False, f"Potential look-ahead detected: fillna(method='{method}')"

            # numpy roll with negative shift is equivalent to future shift.
            if isinstance(func, ast.Attribute) and func.attr == "roll":
                base = func.value
                base_name = base.id if isinstance(base, ast.Name) else None
                if base_name in {"np", "numpy"}:
                    if len(node.args) >= 2 and _is_negative_number(node.args[1]):
                        return False, "Potential look-ahead detected: np.roll() with negative shift"
                    for kw in node.keywords or []:
                        if kw.arg == "shift" and _is_negative_number(kw.value):
                            return False, "Potential look-ahead detected: np.roll() with negative shift"

        # Check class inheritance
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "IStrategy":
                    has_istrategy_base = True
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name in _REQUIRED_METHODS:
                        found_methods.add(item.name)

    if not has_istrategy_base:
        return False, "Strategy class must inherit from IStrategy"

    missing = _REQUIRED_METHODS - found_methods
    if missing:
        return False, f"Missing required methods: {', '.join(sorted(missing))}"

    return True, "Validation passed"


def prepare_sandbox(
    config: "MinerConfig",
    run_dir: Path,
    iteration: int,
    *,
    variant: str | None = None,
) -> Path:
    """Create an isolated sandbox directory for strategy generation.

    Returns the sandbox root path.

    If *variant* is provided, the sandbox is placed under:
      iter_<n>/<variant>/sandbox
    Otherwise (legacy):
      iter_<n>/sandbox
    """

    iter_dir = run_dir / f"iter_{iteration}"
    sandbox = (iter_dir / str(variant) / "sandbox") if variant else (iter_dir / "sandbox")

    strategies_dir = sandbox / "user_data" / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    # Symlink to real OHLCV data
    real_data = paths.user_data_root() / "data"
    sandbox_data = sandbox / "user_data" / "data"
    if real_data.exists() and not sandbox_data.exists():
        sandbox_data.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(real_data.resolve()), str(sandbox_data))

    # Copy FreqTrade config
    ft_config = paths.resolve_repo_path(config.freqtrade_config)
    if ft_config.exists():
        dest = sandbox / ft_config.name
        if not dest.exists():
            shutil.copy2(str(ft_config), str(dest))

    # Copy reference strategy
    ref_strategy = paths.user_data_root() / "strategies" / "ExpressionLongStrategy.py"
    if ref_strategy.exists():
        ref_dest = strategies_dir / "ExpressionLongStrategy_reference.py"
        if not ref_dest.exists():
            shutil.copy2(str(ref_strategy), str(ref_dest))

    # Create backtest results dir
    (sandbox / "user_data" / "backtest_results").mkdir(parents=True, exist_ok=True)

    logger.info("Sandbox prepared at %s", sandbox)
    return sandbox


def find_strategy_files(sandbox: Path) -> list[Path]:
    """Find all .py strategy files in the sandbox strategies dir."""

    strategies_dir = sandbox / "user_data" / "strategies"
    if not strategies_dir.exists():
        return []
    return [
        p
        for p in sorted(strategies_dir.glob("*.py"))
        if not p.name.startswith("_") and "reference" not in p.name.lower()
    ]

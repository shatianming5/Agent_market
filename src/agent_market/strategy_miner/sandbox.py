"""Strategy code validation and sandbox preparation."""
from __future__ import annotations

import ast
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

from agent_market import paths

logger = logging.getLogger(__name__)

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "socket", "requests", "urllib",
    "http", "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "shutil", "pathlib",
})

_FORBIDDEN_CALLS = frozenset({
    "exec", "eval", "open", "__import__", "compile",
    "getattr", "setattr", "delattr", "globals", "locals",
})

_REQUIRED_METHODS = frozenset({
    "populate_indicators",
    "populate_entry_trend",
    "populate_exit_trend",
})


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

        # Check forbidden calls
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in _FORBIDDEN_CALLS:
                return False, f"Forbidden call: {name}()"

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
) -> Path:
    """Create an isolated sandbox directory for strategy generation.

    Returns the sandbox root path.
    """
    sandbox = run_dir / f"iter_{iteration}" / "sandbox"
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
        p for p in sorted(strategies_dir.glob("*.py"))
        if not p.name.startswith("_") and "reference" not in p.name.lower()
    ]

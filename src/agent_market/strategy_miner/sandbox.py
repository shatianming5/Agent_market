"""Strategy code validation and sandbox preparation."""
from __future__ import annotations

import ast
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Tuple

from agent_market import paths

logger = logging.getLogger(__name__)

_FREQTRADE_REQUIRED_ORDERTYPES = ("entry", "exit", "stoploss", "stoploss_on_exchange")
_FREQTRADE_REQUIRED_ORDERTIF = ("entry", "exit")

_DEFAULT_ORDER_TYPES: dict[str, object] = {
    "entry": "market",
    "exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": False,
}
_DEFAULT_ORDER_TIME_IN_FORCE: dict[str, str] = {
    "entry": "GTC",
    "exit": "GTC",
}

_FORBIDDEN_IMPORTS = frozenset(
    {
        "os",
        "subprocess",
        "socket",
        "requests",
        "importlib",
        "ctypes",
        "multiprocessing",
        "signal",
        "pty",
        "webbrowser",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "xmlrpc",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "shutil",
        "pathlib",
        "talib",
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
_DATE_SETINDEX_RE = re.compile(
    r"^(?P<indent>\s*)(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\s*\.set_index\(\s*['\"]date['\"]\s*,\s*inplace\s*=\s*True\s*\)\s*$",
    re.MULTILINE,
)


def _auto_fix_informative_merge_usage(code: str) -> tuple[str, list[str]]:
    if "merge_informative_pair" not in code:
        return code, []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    class _Transformer(ast.NodeTransformer):
        def __init__(self) -> None:
            self.replaced_self_call = False
            self.added_append_timeframe_false = False
            self.saw_merge_call = False

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "merge_informative_pair"
            ):
                node.func = ast.Name(id="merge_informative_pair", ctx=ast.Load())
                self.replaced_self_call = True

            if isinstance(node.func, ast.Name) and node.func.id == "merge_informative_pair":
                self.saw_merge_call = True
                has_suffix = any(kw.arg == "suffix" for kw in node.keywords or [])
                append_kw = next((kw for kw in node.keywords or [] if kw.arg == "append_timeframe"), None)
                if has_suffix and append_kw is None:
                    node.keywords.append(ast.keyword(arg="append_timeframe", value=ast.Constant(False)))
                    self.added_append_timeframe_false = True
            return node

    transformer = _Transformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    fixes: list[str] = []
    if transformer.replaced_self_call:
        fixes.append("rewrite_self_merge_informative_pair")
    if transformer.added_append_timeframe_false:
        fixes.append("force_append_timeframe_false_for_suffix")

    if transformer.saw_merge_call:
        import_added = False
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "freqtrade.strategy":
                imported_names = {alias.name for alias in node.names}
                if "merge_informative_pair" not in imported_names:
                    node.names.append(ast.alias(name="merge_informative_pair", asname=None))
                    import_added = True
                break
        else:
            insert_idx = 0
            while (
                insert_idx < len(tree.body)
                and isinstance(tree.body[insert_idx], ast.ImportFrom)
                and tree.body[insert_idx].module == "__future__"
            ):
                insert_idx += 1
            tree.body.insert(
                insert_idx,
                ast.ImportFrom(
                    module="freqtrade.strategy",
                    names=[ast.alias(name="merge_informative_pair", asname=None)],
                    level=0,
                ),
            )
            import_added = True
        if import_added:
            fixes.append("ensure_merge_informative_pair_import")

    if not fixes:
        return code, []

    try:
        updated = ast.unparse(tree).strip() + "\n"
    except Exception:
        return code, []
    return updated, fixes


def _validate_informative_merge_usage(tree: ast.AST) -> tuple[bool, str]:
    imported_merge = False
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ImportFrom) and node.module == "freqtrade.strategy":
            if any(alias.name == "merge_informative_pair" for alias in node.names):
                imported_merge = True
                break

    saw_merge_call = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "merge_informative_pair"
        ):
            return False, (
                "Use merge_informative_pair(...) imported from freqtrade.strategy; "
                "do not call self.merge_informative_pair(...)"
            )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "merge_informative_pair":
                saw_merge_call = True
                suffix_kw = next((kw for kw in node.keywords or [] if kw.arg == "suffix"), None)
                append_kw = next((kw for kw in node.keywords or [] if kw.arg == "append_timeframe"), None)
                if suffix_kw is not None:
                    if append_kw is None:
                        return False, (
                            "merge_informative_pair with suffix requires append_timeframe=False"
                        )
                    if not (
                        isinstance(append_kw.value, ast.Constant)
                        and append_kw.value.value is False
                    ):
                        return False, (
                            "merge_informative_pair with suffix requires append_timeframe=False"
                        )

    if saw_merge_call and not imported_merge:
        return False, "merge_informative_pair must be imported from freqtrade.strategy"
    return True, "Validation passed"


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
    if "DataFrame" in out or "Order" in out or "Trade" in out:
        future_line = "from __future__ import annotations"
        lines2 = out.splitlines()
        future_idxs = [i for i, l in enumerate(lines2) if l.strip() == future_line]

        if not future_idxs:
            out = future_line + "\n\n" + out.lstrip("\n")
            fixes.append("add_future_annotations")
        else:
            # Normalize: ensure the future import is *once* and at the top.
            prefix: list[str] = []
            rest = list(lines2)
            if rest and rest[0].startswith("#!"):
                prefix.append(rest[0])
                rest = rest[1:]
            if rest and re.match(r"^#.*coding[:=]\s*[-\w.]+", rest[0]):
                prefix.append(rest[0])
                rest = rest[1:]

            rest_wo_future = [l for l in rest if l.strip() != future_line]
            out_norm = "\n".join(prefix + [future_line, ""] + rest_wo_future).strip() + "\n"
            if out_norm != out:
                out = out_norm
                fixes.append("normalize_future_annotations")

    # 5) Common inheritance mismatch: class Foo(Strategy) -> class Foo(IStrategy)
    if "IStrategy" in out and "class" in out:
        # Only replace simple `(Strategy)` occurrences.
        updated = re.sub(r"\(\s*Strategy\s*\)", "(IStrategy)", out)
        if updated != out:
            out = updated
            fixes.append("fix_inheritance_strategy_to_istrategy")

    # 6) Replace talib imports with pandas_ta (talib is not installed).
    if re.search(r"^\s*(import\s+talib|from\s+talib)", out, re.MULTILINE):
        # Replace common talib import patterns
        out = re.sub(
            r"^\s*import\s+talib\.abstract\s+as\s+(\w+)\s*$",
            r"import pandas_ta as \1",
            out,
            flags=re.MULTILINE,
        )
        out = re.sub(
            r"^\s*import\s+talib\s+as\s+(\w+)\s*$",
            r"import pandas_ta as \1",
            out,
            flags=re.MULTILINE,
        )
        out = re.sub(
            r"^\s*from\s+talib\.abstract\s+import\s+.*$",
            "import pandas_ta as ta",
            out,
            flags=re.MULTILINE,
        )
        out = re.sub(
            r"^\s*from\s+talib\s+import\s+.*$",
            "import pandas_ta as ta",
            out,
            flags=re.MULTILINE,
        )
        fixes.append("replace_talib_with_pandas_ta")

    # 7) Normalize informative merge usage.
    rewritten_merge, merge_fixes = _auto_fix_informative_merge_usage(out)
    if merge_fixes:
        out = rewritten_merge
        fixes.extend(merge_fixes)

    # 8) Preserve the `date` column for merge_informative_pair while still
    # allowing strategies to work with a DatetimeIndex for VWAP/session logic.
    if "merge_informative_pair" in out and re.search(r"^\s*import\s+pandas\s+as\s+pd\s*$", out, re.MULTILINE):
        def _rewrite_set_index(match: re.Match[str]) -> str:
            indent = match.group("indent")
            obj = match.group("obj")
            return (
                f'{indent}if "date" in {obj}.columns:\n'
                f'{indent}    {obj}["date"] = pd.to_datetime({obj}["date"], utc=True, errors="coerce")\n'
                f'{indent}    {obj}.index = pd.DatetimeIndex({obj}["date"])'
            )

        rewritten = _DATE_SETINDEX_RE.sub(_rewrite_set_index, out)
        if rewritten != out:
            out = rewritten
            fixes.append("preserve_date_column_for_informative_merge")

    # 9) Freqtrade hyperopt Parameters expose `.value`, not `.default`.
    # Some LLM outputs use `.default` and crash at runtime.
    rewritten = re.sub(r"(\bself\.[A-Za-z_]\w*)\.default\b", r"\1.value", out)
    if rewritten != out:
        out = rewritten
        fixes.append("rewrite_parameter_default_to_value")

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


def ensure_freqtrade_strategy_compliance_code(
    code: str,
    *,
    timeframe: str = "1h",
    enforce_can_short_false: bool = True,
) -> tuple[str, list[str]]:
    """Ensure generated strategies pass freqtrade sanity checks (best-effort).

    This patches the IStrategy class body to ensure:
    - timeframe is set (and optionally forced to *timeframe*)
    - order_types includes required keys
    - order_time_in_force includes required keys
    - can_short is False in spot-mode pipelines (optional)

    It does NOT import freqtrade at runtime.
    """
    if not isinstance(code, str) or not code.strip():
        return "", ["non_string_input"]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    fixes: list[str] = []

    def _is_istrategy_base(base: ast.expr) -> bool:
        if isinstance(base, ast.Name):
            return base.id == "IStrategy"
        if isinstance(base, ast.Attribute):
            return base.attr == "IStrategy"
        return False

    def _targets_name(stmt: ast.stmt, name: str) -> bool:
        if isinstance(stmt, ast.Assign):
            return any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
        if isinstance(stmt, ast.AnnAssign):
            return isinstance(stmt.target, ast.Name) and stmt.target.id == name
        return False

    def _set_value(stmt: ast.stmt, value: ast.expr) -> bool:
        if isinstance(stmt, ast.Assign):
            stmt.value = value
            return True
        if isinstance(stmt, ast.AnnAssign):
            stmt.value = value
            return True
        return False

    def _dict_literal_keys(expr: ast.expr) -> set[str] | None:
        if not isinstance(expr, ast.Dict):
            return None
        keys: set[str] = set()
        for k in expr.keys:
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                return None
            keys.add(k.value)
        return keys

    def _make_dict(mapping: dict[str, object]) -> ast.Dict:
        keys = [ast.Constant(k) for k in mapping.keys()]
        vals: list[ast.expr] = []
        for v in mapping.values():
            if isinstance(v, bool):
                vals.append(ast.Constant(v))
            elif isinstance(v, (int, float, str)):
                vals.append(ast.Constant(v))
            else:
                vals.append(ast.Constant(str(v)))
        return ast.Dict(keys=keys, values=vals)

    def _inject_ohlcv_suffix_guard(fn: ast.FunctionDef) -> bool:
        """Ensure required OHLCV columns exist after informative merges.

        Some generated strategies use `pd.merge_asof` and accidentally bring an
        informative `close` column, which results in suffixing to `close_x` /
        `close_y` and breaks freqtrade validation (expects `close`).
        """
        # Idempotent: if the function already references close_x, assume guard exists.
        for n in ast.walk(fn):
            if isinstance(n, ast.Constant) and n.value == "close_x":
                return False

        df_name = "dataframe"
        try:
            # populate_indicators(self, dataframe, metadata, ...)
            if len(fn.args.args) >= 2:
                df_name = str(fn.args.args[1].arg or "dataframe")
        except Exception:
            df_name = "dataframe"

        cols = ("open", "high", "low", "close", "volume")
        guard: list[ast.stmt] = []
        for c in cols:
            cx = f"{c}_x"
            cond = ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Constant(c),
                        ops=[ast.NotIn()],
                        comparators=[ast.Attribute(value=ast.Name(id=df_name, ctx=ast.Load()), attr="columns", ctx=ast.Load())],
                    ),
                    ast.Compare(
                        left=ast.Constant(cx),
                        ops=[ast.In()],
                        comparators=[ast.Attribute(value=ast.Name(id=df_name, ctx=ast.Load()), attr="columns", ctx=ast.Load())],
                    ),
                ],
            )
            assign = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id=df_name, ctx=ast.Load()),
                        slice=ast.Constant(c),
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Subscript(
                    value=ast.Name(id=df_name, ctx=ast.Load()),
                    slice=ast.Constant(cx),
                    ctx=ast.Load(),
                ),
            )
            guard.append(ast.If(test=cond, body=[assign], orelse=[]))

        # Insert guard right before the final `return ...` if present.
        last_ret = None
        for i, stmt in enumerate(fn.body):
            if isinstance(stmt, ast.Return):
                last_ret = i
        if last_ret is None:
            return False

        fn.body[last_ret:last_ret] = guard
        return True

    class _Transformer(ast.NodeTransformer):
        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            is_target = any(_is_istrategy_base(b) for b in node.bases)
            if not is_target:
                return self.generic_visit(node)

            found_timeframe = False
            found_order_types = False
            found_order_tif = False
            found_can_short = False
            found_stoploss = False
            found_minimal_roi = False

            for stmt in node.body:
                if _targets_name(stmt, "timeframe"):
                    found_timeframe = True
                    if timeframe:
                        v = getattr(stmt, "value", None)
                        if not (isinstance(v, ast.Constant) and v.value == timeframe):
                            _set_value(stmt, ast.Constant(timeframe))
                            fixes.append("fix_timeframe")
                elif _targets_name(stmt, "order_types"):
                    found_order_types = True
                    v = getattr(stmt, "value", None)
                    keys = _dict_literal_keys(v) if isinstance(v, ast.expr) else None
                    needs_fix = keys is None or not all(
                        k in keys for k in _FREQTRADE_REQUIRED_ORDERTYPES
                    )
                    if not needs_fix and isinstance(v, ast.Dict):
                        idx_by_key: dict[str, int] = {}
                        for i, (k_node, _v_node) in enumerate(zip(v.keys, v.values)):
                            if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                                idx_by_key[k_node.value] = i

                        allowed = {"limit", "market"}
                        for k_req in ("entry", "exit", "stoploss"):
                            i = idx_by_key.get(k_req)
                            if i is None:
                                needs_fix = True
                                break
                            v_node = v.values[i]
                            if not (isinstance(v_node, ast.Constant) and isinstance(v_node.value, str)):
                                needs_fix = True
                                break
                            norm = v_node.value.strip().lower()
                            if norm not in allowed:
                                needs_fix = True
                                break
                            if v_node.value != norm:
                                v.values[i] = ast.Constant(norm)
                                fixes.append("normalize_order_types_values")

                        if not needs_fix:
                            i = idx_by_key.get("stoploss_on_exchange")
                            if i is None:
                                needs_fix = True
                            else:
                                v_node = v.values[i]
                                if not (
                                    isinstance(v_node, ast.Constant)
                                    and isinstance(v_node.value, bool)
                                ):
                                    v.values[i] = ast.Constant(False)
                                    fixes.append("normalize_order_types_values")

                    if needs_fix:
                        _set_value(stmt, _make_dict(_DEFAULT_ORDER_TYPES))
                        fixes.append("fix_order_types")
                elif _targets_name(stmt, "order_time_in_force"):
                    found_order_tif = True
                    v = getattr(stmt, "value", None)
                    keys = _dict_literal_keys(v) if isinstance(v, ast.expr) else None
                    needs_fix = keys is None or not all(k in keys for k in _FREQTRADE_REQUIRED_ORDERTIF)
                    if not needs_fix and isinstance(v, ast.Dict):
                        idx_by_key: dict[str, int] = {}
                        for i, (k_node, _v_node) in enumerate(zip(v.keys, v.values)):
                            if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                                idx_by_key[k_node.value] = i

                        for k_req in ("entry", "exit"):
                            i = idx_by_key.get(k_req)
                            if i is None:
                                needs_fix = True
                                break
                            v_node = v.values[i]
                            if not (isinstance(v_node, ast.Constant) and isinstance(v_node.value, str)):
                                needs_fix = True
                                break
                            norm = v_node.value.strip().upper()
                            if not norm:
                                needs_fix = True
                                break
                            if v_node.value != norm:
                                v.values[i] = ast.Constant(norm)
                                fixes.append("normalize_order_tif_values")

                    if needs_fix:
                        _set_value(stmt, _make_dict(_DEFAULT_ORDER_TIME_IN_FORCE))
                        fixes.append("fix_order_time_in_force")
                elif _targets_name(stmt, "can_short"):
                    found_can_short = True
                    if enforce_can_short_false:
                        v = getattr(stmt, "value", None)
                        if not (isinstance(v, ast.Constant) and v.value is False):
                            _set_value(stmt, ast.Constant(False))
                            fixes.append("fix_can_short_false")
                elif _targets_name(stmt, "stoploss"):
                    found_stoploss = True
                    v = getattr(stmt, "value", None)
                    # Freqtrade expects a numeric stoploss (float). Some LLM outputs
                    # incorrectly assign a HyperOpt Parameter here (DecimalParameter),
                    # which crashes the resolver.
                    is_numeric = isinstance(v, ast.Constant) and isinstance(v.value, (int, float))
                    if not is_numeric:
                        _set_value(stmt, ast.Constant(-0.02))
                        fixes.append("fix_stoploss")
                elif _targets_name(stmt, "minimal_roi"):
                    found_minimal_roi = True
                    v = getattr(stmt, "value", None)
                    if not isinstance(v, ast.Dict):
                        _set_value(stmt, _make_dict({"0": 0.01, "30": 0.005, "120": 0.0}))
                        fixes.append("fix_minimal_roi")

            insert_at = 0
            if node.body and isinstance(node.body[0], ast.Expr):
                v0 = getattr(node.body[0], "value", None)
                if isinstance(v0, ast.Constant) and isinstance(v0.value, str):
                    insert_at = 1

            if not found_can_short and enforce_can_short_false:
                node.body.insert(
                    insert_at,
                    ast.Assign(
                        targets=[ast.Name(id="can_short", ctx=ast.Store())],
                        value=ast.Constant(False),
                    ),
                )
                insert_at += 1
                fixes.append("add_can_short_false")

            if not found_stoploss:
                node.body.insert(
                    insert_at,
                    ast.Assign(
                        targets=[ast.Name(id="stoploss", ctx=ast.Store())],
                        value=ast.Constant(-0.02),
                    ),
                )
                insert_at += 1
                fixes.append("add_stoploss")

            if not found_minimal_roi:
                node.body.insert(
                    insert_at,
                    ast.Assign(
                        targets=[ast.Name(id="minimal_roi", ctx=ast.Store())],
                        value=_make_dict({"0": 0.01, "30": 0.005, "120": 0.0}),
                    ),
                )
                insert_at += 1
                fixes.append("add_minimal_roi")

            if not found_timeframe and timeframe:
                node.body.insert(
                    insert_at,
                    ast.Assign(
                        targets=[ast.Name(id="timeframe", ctx=ast.Store())],
                        value=ast.Constant(timeframe),
                    ),
                )
                insert_at += 1
                fixes.append("add_timeframe")

            if not found_order_types:
                node.body.insert(
                    insert_at,
                    ast.Assign(
                        targets=[ast.Name(id="order_types", ctx=ast.Store())],
                        value=_make_dict(_DEFAULT_ORDER_TYPES),
                    ),
                )
                insert_at += 1
                fixes.append("add_order_types")

            if not found_order_tif:
                node.body.insert(
                    insert_at,
                    ast.Assign(
                        targets=[ast.Name(id="order_time_in_force", ctx=ast.Store())],
                        value=_make_dict(_DEFAULT_ORDER_TIME_IN_FORCE),
                    ),
                )
                fixes.append("add_order_time_in_force")

            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "populate_indicators":
                    if _inject_ohlcv_suffix_guard(stmt):
                        fixes.append("add_ohlcv_suffix_guard")

            return self.generic_visit(node)

    _Transformer().visit(tree)
    if not fixes:
        return code, []

    try:
        ast.fix_missing_locations(tree)
        out = ast.unparse(tree).strip() + "\n"
        return out, fixes
    except Exception:
        return code, []


def ensure_freqtrade_strategy_compliance_file(
    path: Path,
    *,
    timeframe: str = "1h",
    enforce_can_short_false: bool = True,
) -> tuple[bool, list[str]]:
    """Apply `ensure_freqtrade_strategy_compliance_code` to an on-disk strategy file."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False, ["read_failed"]

    fixed, fixes = ensure_freqtrade_strategy_compliance_code(
        raw,
        timeframe=timeframe,
        enforce_can_short_false=enforce_can_short_false,
    )
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

        # Check forbidden attribute access patterns (__builtins__, sys.modules, etc.)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in ("__builtins__", "__loader__"):
                return False, f"Forbidden access: {node.value.id}[...]"
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr == "modules":
                return False, "Forbidden access: sys.modules"
            if node.attr in ("__builtins__", "__loader__", "__subclasses__"):
                return False, f"Forbidden attribute: .{node.attr}"

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
            # Block breakpoint()
            if isinstance(func, ast.Name) and func.id == "breakpoint":
                return False, "Forbidden call: breakpoint()"

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

    if "merge_informative_pair" in code and _DATE_SETINDEX_RE.search(code):
        return False, (
            "merge_informative_pair requires a preserved 'date' column; "
            "do not use set_index('date', inplace=True) before merging"
        )

    merge_ok, merge_msg = _validate_informative_merge_usage(tree)
    if not merge_ok:
        return False, merge_msg

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

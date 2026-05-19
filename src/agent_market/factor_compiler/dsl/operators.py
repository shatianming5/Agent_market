from __future__ import annotations

from ..api_models import ExprArg, ExprNode
from .ast_utils import literal_int, format_number
from .grammar import (
    CONST_OP,
    OP_TO_BINOP_SYMBOL,
    OP_TO_CMP_SYMBOL,
    OP_TO_UNARY_SYMBOL,
    VAR_OP,
    is_safe_identifier,
)


class OperatorCompileError(ValueError):
    pass


CALL_ALIASES: dict[str, str] = {
    "rolling_mean": "roll_mean",
    "rolling_std": "roll_std",
    "lag": "shift",
}

_MICROSTRUCTURE_COLUMN_OPS = {"mid", "spread", "microprice", "trade_sign"}
_MICROSTRUCTURE_WINDOW_OPS = {"vwap", "ofi", "arrival_intensity", "rv"}
_MICROSTRUCTURE_LEVEL_OPS = {"depth_bid", "depth_ask"}


def _map_call(op: str, args: list[ExprArg]) -> str:
    if op == "zscore":
        if len(args) == 1:
            return "z"
        if len(args) == 2:
            return "ts_z"
        raise OperatorCompileError("zscore expects 1 or 2 args")
    return CALL_ALIASES.get(op, op)

def _depth_suffix(arg: ExprArg, *, prefix: str) -> int | None:
    if isinstance(arg, ExprNode):
        op = str(arg.op)
        args = list(arg.args or [])
        if op == prefix and len(args) == 1:
            levels = literal_int(args[0])
            if levels is None or levels <= 0:
                return None
            return int(levels)
        if op == VAR_OP and len(args) == 1 and isinstance(args[0], str):
            name = str(args[0]).strip()
            key = f"{prefix}_"
            if name.startswith(key):
                suf = name[len(key) :]
                if suf.isdigit():
                    levels = int(suf)
                    return levels if levels > 0 else None
    return None


def _compile_microstructure_call(op: str, args: list[ExprArg]) -> str | None:
    """
    Compile a small set of microstructure operators into existing feature column names.

    This keeps ExpressionEngine unchanged by avoiding adding new callables.
    """

    if op in _MICROSTRUCTURE_COLUMN_OPS:
        expected = {"mid": (0, 2), "spread": (0, 2), "microprice": (0, 4), "trade_sign": (0,)}
        if op in expected and len(args) not in expected[op]:
            raise OperatorCompileError(f"{op} expects {expected[op]} args")
        return op

    if op in _MICROSTRUCTURE_LEVEL_OPS:
        if len(args) != 1:
            raise OperatorCompileError(f"{op} expects 1 arg")
        levels = literal_int(args[0])
        if levels is None:
            raise OperatorCompileError(f"{op} levels must be an integer constant")
        if levels <= 0:
            raise OperatorCompileError(f"{op} levels must be > 0")
        return f"{op}_{int(levels)}"

    if op == "imbalance":
        if len(args) != 2:
            raise OperatorCompileError("imbalance expects 2 args")
        bid_levels = _depth_suffix(args[0], prefix="depth_bid")
        ask_levels = _depth_suffix(args[1], prefix="depth_ask")
        if bid_levels is not None and ask_levels is not None and bid_levels == ask_levels:
            return f"imbalance_{int(bid_levels)}"

        left = _compile(args[0])
        right = _compile(args[1])
        # Common definition: (bid-ask)/(bid+ask) with eps guard.
        return f"(({left}-{right})/(({left}+{right})+1e-12))"

    if op in _MICROSTRUCTURE_WINDOW_OPS:
        if len(args) != 1:
            raise OperatorCompileError(f"{op} expects 1 arg")
        w = literal_int(args[0])
        if w is None:
            raise OperatorCompileError(f"{op} window must be an integer constant")
        if w <= 0:
            raise OperatorCompileError(f"{op} window must be > 0")
        return f"{op}_{int(w)}"

    return None


def _compile(arg: ExprArg) -> str:
    if isinstance(arg, ExprNode):
        op = str(arg.op)
        args = list(arg.args or [])

        if op == VAR_OP:
            if len(args) != 1 or not isinstance(args[0], str) or not is_safe_identifier(args[0]):
                raise OperatorCompileError(f"Invalid var node: {arg!r}")
            return args[0]

        if op == CONST_OP:
            if len(args) != 1:
                raise OperatorCompileError(f"Invalid const node: {arg!r}")
            return format_number(args[0])

        if op in OP_TO_BINOP_SYMBOL:
            if len(args) != 2:
                raise OperatorCompileError(f"Invalid binary node: {arg!r}")
            left = _compile(args[0])
            right = _compile(args[1])
            return f"({left}{OP_TO_BINOP_SYMBOL[op]}{right})"

        if op in OP_TO_UNARY_SYMBOL:
            if len(args) != 1:
                raise OperatorCompileError(f"Invalid unary node: {arg!r}")
            inner = _compile(args[0])
            return f"({OP_TO_UNARY_SYMBOL[op]}{inner})"

        if op in OP_TO_CMP_SYMBOL:
            if len(args) != 2:
                raise OperatorCompileError(f"Invalid compare node: {arg!r}")
            left = _compile(args[0])
            right = _compile(args[1])
            return f"({left}{OP_TO_CMP_SYMBOL[op]}{right})"

        micro = _compile_microstructure_call(op, args)
        if micro is not None:
            return micro

        mapped = _map_call(op, args)
        if not is_safe_identifier(mapped):
            raise OperatorCompileError(f"Invalid call name: {mapped!r}")
        inner = ",".join(_compile(a) for a in args)
        return f"{mapped}({inner})"

    return format_number(arg)


def compile_to_expression_engine(expr: ExprNode) -> str:
    """
    Compile Factor Compiler canonical ExprNode into an ExpressionEngine-evaluable expression string.

    This performs small operator aliasing (e.g. rolling_mean->roll_mean, zscore->ts_z) but does not
    attempt to implement microstructure-specific operators yet.
    """

    return _compile(expr)

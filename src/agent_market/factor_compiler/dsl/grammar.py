from __future__ import annotations

import ast
import re
from typing import Dict, Tuple, Type

MAX_FORMULA_LENGTH = 1024
MAX_AST_NODES = 512

# Match a Python identifier (restricted) and keep it compatible with ExpressionEngine.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def is_safe_identifier(name: str) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if name.startswith("_"):
        return False
    return _IDENT_RE.match(name) is not None


BINOP_TO_OP: Dict[Type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.FloorDiv: "floordiv",
}

OP_TO_BINOP_SYMBOL: Dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "pow": "**",
    "floordiv": "//",
}

UNARYOP_TO_OP: Dict[Type[ast.unaryop], str] = {
    ast.UAdd: "pos",
    ast.USub: "neg",
}

OP_TO_UNARY_SYMBOL: Dict[str, str] = {
    "pos": "+",
    "neg": "-",
}

CMPOP_TO_OP: Dict[Type[ast.cmpop], str] = {
    ast.Gt: "gt",
    ast.GtE: "gte",
    ast.Lt: "lt",
    ast.LtE: "lte",
    ast.Eq: "eq",
    ast.NotEq: "neq",
}

OP_TO_CMP_SYMBOL: Dict[str, str] = {
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "eq": "==",
    "neq": "!=",
}

# Internal node ops used by the DSL
VAR_OP = "var"
CONST_OP = "const"


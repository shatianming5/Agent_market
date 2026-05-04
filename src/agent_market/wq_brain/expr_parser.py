"""Recursive-descent FASTEXPR validator (no execution).

Inspired by QuantGPT's expression_parser.py but rebuilt as a slim validation-
only subset (no pandas dependency, no DataFrame ops). Catches syntax errors,
arity mismatches, nesting overflow, unknown ops/fields, and forbidden Python
syntax — all the cases our previous token-only validator leaked through.

Supports:
- Function calls with positional args:   `ts_mean(close, 20)`
- Arithmetic + unary minus:              `(close - vwap) / adv60`
- Multi-statement bindings via `;`:      `x = ts_mean(close,20); rank(close - x)`
- Numeric literals (int / float):        `rank(close, 0.5)`
- Identifiers (fields / variables / op names)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .operators import (
    ALL_FIELDS,
    ALL_OPERATORS,
    FIELDS_FUNDAMENTAL_UNAVAILABLE,
    OPERATORS_TS_UNAVAILABLE,
)

# ── Hard limits ──────────────────────────────────────────────────────────
MAX_EXPRESSION_LENGTH = 500
MAX_NESTING_DEPTH = 10
MAX_STATEMENTS = 10

# ── Operator arity metadata ─────────────────────────────────────────────
# (min_args, max_args) per operator. None for max_args = unbounded.
_OP_META: dict[str, tuple[int, Optional[int]]] = {
    # Time-series: most are (field, window) → arity 2
    "ts_mean": (2, 2),
    "ts_rank": (2, 2),
    "ts_zscore": (2, 2),
    "ts_sum": (2, 2),
    "ts_max": (2, 2),
    "ts_delta": (2, 2),
    "ts_delay": (2, 2),
    "ts_decay_linear": (2, 2),
    "ts_corr": (3, 3),  # (x, y, window)
    # Cross-sectional
    "rank": (1, 1),
    "group_rank": (2, 2),
    "group_mean": (2, 2),
    "group_sum": (2, 2),
    "group_std": (2, 2),
    "group_zscore": (2, 2),
    "group_neutralize": (2, 2),
    "group_count": (2, 2),
    "scale": (1, 2),  # optional second arg for scale factor
    "signed_power": (2, 2),
    # Math
    "abs": (1, 1),
    "log": (1, 1),
    "sqrt": (1, 1),
    "sign": (1, 1),
    "power": (2, 2),
    "exp": (1, 1),
    "min": (2, None),  # variadic: min(a,b) or min(a,b,c)
    "max": (2, None),
    "sum": (1, 1),
    "mean": (1, None),
    "if_else": (3, 3),
    "clamp": (3, 3),
    "correlation": (3, 3),
    "covariance": (3, 3),
    # Turnover-reduction transform — WQ accepts ONLY 1 arg in our tier
    # (`hump(alpha, 0.01)` → "Invalid number of inputs: 2")
    "hump": (1, 1),
}

_TOKEN_RE = re.compile(
    r"\s+"
    r"|(?P<NUMBER>\d+(?:\.\d+)?)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<OP>==|!=|<=|>=|&&|\|\||[+\-*/<>=,();])"
)
_PYTHON_FORBIDDEN = ("import ", "def ", "class ", "lambda", "print(",
                     "exec(", "eval(", "__")


@dataclass
class Token:
    kind: str  # NUMBER | IDENT | OP | EOF
    value: str
    pos: int


@dataclass
class Node:
    kind: str  # number | ident | call | binop | unaryop | assign | program
    value: str = ""
    args: list["Node"] = field(default_factory=list)


class ParseError(Exception):
    def __init__(self, message: str, pos: int = -1) -> None:
        super().__init__(message)
        self.pos = pos


def tokenize(expr: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(expr):
        m = _TOKEN_RE.match(expr, i)
        if not m:
            raise ParseError(f"unexpected character {expr[i]!r}", i)
        if m.lastgroup is None:  # whitespace
            i = m.end()
            continue
        tokens.append(Token(kind=m.lastgroup, value=m.group(), pos=i))
        i = m.end()
    tokens.append(Token(kind="EOF", value="", pos=len(expr)))
    return tokens


class Parser:
    """Recursive-descent parser. Builds an AST without evaluating."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0
        self.depth = 0
        self.max_depth = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def eat(self, kind: str, value: Optional[str] = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (value is not None and tok.value != value):
            expected = f"{kind}={value!r}" if value else kind
            raise ParseError(f"expected {expected}, got {tok.kind}={tok.value!r}", tok.pos)
        self.i += 1
        return tok

    def parse_program(self) -> Node:
        """program := stmt (';' stmt)*"""
        stmts = [self.parse_stmt()]
        while self.peek().kind == "OP" and self.peek().value == ";":
            self.eat("OP", ";")
            if self.peek().kind == "EOF":
                break
            stmts.append(self.parse_stmt())
        if self.peek().kind != "EOF":
            tok = self.peek()
            raise ParseError(f"trailing token {tok.value!r}", tok.pos)
        if len(stmts) > MAX_STATEMENTS:
            raise ParseError(f"too many statements ({len(stmts)} > {MAX_STATEMENTS})", -1)
        return Node(kind="program", args=stmts)

    def parse_stmt(self) -> Node:
        """stmt := ident '=' expr | expr"""
        # Lookahead: is this a binding `name = ...`?
        if (self.peek().kind == "IDENT"
                and self.i + 1 < len(self.tokens)
                and self.tokens[self.i + 1].kind == "OP"
                and self.tokens[self.i + 1].value == "="):
            name_tok = self.eat("IDENT")
            self.eat("OP", "=")
            value = self.parse_expr()
            return Node(kind="assign", value=name_tok.value, args=[value])
        return self.parse_expr()

    def parse_expr(self) -> Node:
        """expr := add ('<'|'<='|'>'|'>='|'=='|'!=' add)*"""
        left = self.parse_add()
        while self.peek().kind == "OP" and self.peek().value in ("<", "<=", ">", ">=", "==", "!="):
            op = self.eat("OP").value
            right = self.parse_add()
            left = Node(kind="binop", value=op, args=[left, right])
        return left

    def parse_add(self) -> Node:
        """add := mul (('+'|'-') mul)*"""
        left = self.parse_mul()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.eat("OP").value
            right = self.parse_mul()
            left = Node(kind="binop", value=op, args=[left, right])
        return left

    def parse_mul(self) -> Node:
        """mul := unary (('*'|'/') unary)*"""
        left = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/"):
            op = self.eat("OP").value
            right = self.parse_unary()
            left = Node(kind="binop", value=op, args=[left, right])
        return left

    def parse_unary(self) -> Node:
        """unary := '-' unary | '+' unary | atom"""
        if self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.eat("OP").value
            inner = self.parse_unary()
            return Node(kind="unaryop", value=op, args=[inner])
        return self.parse_atom()

    def parse_atom(self) -> Node:
        """atom := NUMBER | IDENT '(' args ')' | IDENT | '(' expr ')'"""
        tok = self.peek()
        if tok.kind == "NUMBER":
            self.eat("NUMBER")
            return Node(kind="number", value=tok.value)
        if tok.kind == "OP" and tok.value == "(":
            self.eat("OP", "(")
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            if self.max_depth > MAX_NESTING_DEPTH:
                raise ParseError(
                    f"nesting depth {self.max_depth} exceeds limit {MAX_NESTING_DEPTH}",
                    tok.pos,
                )
            inner = self.parse_expr()
            self.eat("OP", ")")
            self.depth -= 1
            return inner
        if tok.kind == "IDENT":
            self.eat("IDENT")
            # Function call?
            if self.peek().kind == "OP" and self.peek().value == "(":
                self.eat("OP", "(")
                self.depth += 1
                self.max_depth = max(self.max_depth, self.depth)
                if self.max_depth > MAX_NESTING_DEPTH:
                    raise ParseError(
                        f"nesting depth {self.max_depth} exceeds limit {MAX_NESTING_DEPTH}",
                        tok.pos,
                    )
                args: list[Node] = []
                if not (self.peek().kind == "OP" and self.peek().value == ")"):
                    args.append(self.parse_expr())
                    while self.peek().kind == "OP" and self.peek().value == ",":
                        self.eat("OP", ",")
                        args.append(self.parse_expr())
                self.eat("OP", ")")
                self.depth -= 1
                return Node(kind="call", value=tok.value, args=args)
            return Node(kind="ident", value=tok.value)
        raise ParseError(f"unexpected token {tok.kind}={tok.value!r}", tok.pos)


def _walk_check(node: Node, errors: list[str], known_locals: Optional[set[str]] = None) -> None:
    """Post-parse semantic checks."""
    locals_set = known_locals or set()
    if node.kind == "program":
        seen: set[str] = set()
        for stmt in node.args:
            if stmt.kind == "assign":
                seen.add(stmt.value)
                _walk_check(stmt.args[0], errors, seen)
            else:
                _walk_check(stmt, errors, seen)
        return

    if node.kind == "assign":
        _walk_check(node.args[0], errors, locals_set)
        return

    if node.kind == "call":
        op = node.value
        # Operator known?
        if op not in ALL_OPERATORS:
            if op in OPERATORS_TS_UNAVAILABLE:
                errors.append(f"unavailable operator: {op}")
            else:
                errors.append(f"unknown operator: {op}")
        else:
            # Check arity
            meta = _OP_META.get(op)
            if meta is not None:
                min_a, max_a = meta
                n = len(node.args)
                if n < min_a:
                    errors.append(
                        f"{op} requires at least {min_a} args, got {n}"
                    )
                elif max_a is not None and n > max_a:
                    errors.append(
                        f"{op} accepts at most {max_a} args, got {n}"
                    )
        for child in node.args:
            _walk_check(child, errors, locals_set)
        return

    if node.kind == "ident":
        name = node.value
        if name in locals_set:
            return
        if name in ALL_FIELDS:
            return
        if name in FIELDS_FUNDAMENTAL_UNAVAILABLE:
            errors.append(f"unavailable field: {name}")
            return
        # Allow lowercase boolean/keyword-ish identifiers used in conditions
        if name.lower() in {"true", "false"}:
            return
        # Otherwise unknown — could be a var binding the user forgot to define
        errors.append(f"unknown identifier: {name}")
        return

    # binop / unaryop / number — recurse / leaf
    for child in node.args:
        _walk_check(child, errors, locals_set)


def validate_expression_strict(expr: str) -> list[str]:
    """Strict validator: full recursive-descent + semantic checks.

    Returns list of error messages; empty list = OK.
    """
    errors: list[str] = []
    if not expr or not expr.strip():
        return ["empty expression"]
    if len(expr) > MAX_EXPRESSION_LENGTH:
        errors.append(f"expression too long ({len(expr)} > {MAX_EXPRESSION_LENGTH} chars)")
    for kw in _PYTHON_FORBIDDEN:
        if kw in expr:
            errors.append(f"forbidden Python syntax: {kw!r}")
    if errors:
        return errors

    # Quick paren balance check (covers obvious cases before parser)
    if expr.count("(") != expr.count(")"):
        errors.append(
            f"unbalanced parens: {expr.count('(')} '(' vs {expr.count(')')} ')'"
        )
        return errors

    try:
        toks = tokenize(expr)
        parser = Parser(toks)
        ast = parser.parse_program()
    except ParseError as exc:
        errors.append(f"parse error: {exc}")
        return errors

    _walk_check(ast, errors)
    return errors


def parse(expr: str) -> Node:
    """Parse and return AST. Raises ParseError on failure."""
    toks = tokenize(expr)
    return Parser(toks).parse_program()


def count_nesting(expr: str) -> int:
    """Standalone helper: max paren-nesting depth."""
    max_d = d = 0
    for ch in expr:
        if ch == "(":
            d += 1
            max_d = max(max_d, d)
        elif ch == ")":
            d -= 1
    return max_d

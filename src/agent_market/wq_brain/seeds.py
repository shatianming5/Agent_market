"""Seed templates + cartesian-grid expansion for non-LLM batch scan."""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")


@dataclass
class SeedTemplate:
    pattern: str

    def placeholders(self) -> list[str]:
        return sorted(set(_PLACEHOLDER_RE.findall(self.pattern)))

    def expand(self, grid: dict[str, list[Any]]) -> list[str]:
        ph = self.placeholders()
        if not ph:
            return [self.pattern]
        missing = [p for p in ph if p not in grid]
        if missing:
            raise ValueError(f"Template {self.pattern!r}: placeholders {missing} not in grid")
        values = [grid[p] for p in ph]
        out: list[str] = []
        for combo in itertools.product(*values):
            expr = self.pattern
            for placeholder, value in zip(ph, combo):
                expr = expr.replace(placeholder, str(value))
            out.append(expr)
        return out


def load_seeds_config(path: Path) -> tuple[list[SeedTemplate], dict[str, list[Any]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = [SeedTemplate(s) for s in data.get("seeds", [])]
    grid = data.get("grid", {})
    return seeds, grid


def expand_all(seeds: list[SeedTemplate], grid: dict[str, list[Any]]) -> list[str]:
    out: list[str] = []
    for s in seeds:
        out.extend(s.expand(grid))
    return out

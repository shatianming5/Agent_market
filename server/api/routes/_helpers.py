"""Shared helpers for API route handlers."""
from __future__ import annotations

from typing import Optional


def parse_csv(value: Optional[str]) -> list[str]:
    """Parse a comma- or space-separated string into a list of stripped tokens."""
    if not value:
        return []
    raw = str(value).replace(",", " ").split()
    return [x.strip() for x in raw if x.strip()]

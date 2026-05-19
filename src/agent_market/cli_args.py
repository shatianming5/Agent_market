from __future__ import annotations

from typing import Any, Optional


def extract_flag_value(args: Any, flag: str) -> Optional[str]:
    if not isinstance(args, list):
        return None
    value: Optional[str] = None
    for idx, item in enumerate(args):
        if str(item) == flag and idx + 1 < len(args):
            value = str(args[idx + 1])
    return value


def has_flag(args: Any, flag: str) -> bool:
    return isinstance(args, list) and flag in [str(item) for item in args]


__all__ = ["extract_flag_value", "has_flag"]

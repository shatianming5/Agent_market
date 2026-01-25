from __future__ import annotations

import re
from typing import Optional, Tuple

_TIMEFRAME_RE = re.compile(r"^\s*\d+(ms|s|m|h|d|w)\s*$", re.IGNORECASE)


def validate_timeframe(tf: str) -> bool:
    return bool(tf and _TIMEFRAME_RE.match(str(tf)))


def validate_pairs_string(pairs: Optional[str]) -> Tuple[bool, list[str]]:
    if not pairs:
        return True, []
    raw = pairs.replace(",", " ").split()
    ok = []
    for token in raw:
        if re.match(r"^[A-Za-z0-9]+/[A-Za-z0-9]+$", token):
            ok.append(token)
        else:
            return False, []
    return True, ok


__all__ = ["validate_timeframe", "validate_pairs_string"]

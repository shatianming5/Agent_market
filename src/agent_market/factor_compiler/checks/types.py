from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "code": self.code,
            "message": self.message,
            "details": dict(self.details or {}),
        }


def ok(name: str, *, message: str = "ok", details: Optional[Dict[str, Any]] = None) -> CheckResult:
    return CheckResult(name=name, ok=True, code="OK", message=message, details=details or {})


def fail(
    name: str,
    *,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> CheckResult:
    return CheckResult(name=name, ok=False, code=str(code), message=str(message), details=details or {})


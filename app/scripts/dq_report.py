#!/usr/bin/env python
"""Compatibility wrapper for scripts.data.dq_report."""

import sys as _sys
from importlib import import_module as _import_module
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[1]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
_src = _root / "src"
if _src.exists() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

_impl = _import_module("scripts.data.dq_report")
_sys.modules[__name__] = _impl

if __name__ == "__main__":
    _main = getattr(_impl, "main", None)
    if callable(_main):
        _main()

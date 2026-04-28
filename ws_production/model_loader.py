"""Dynamic model loader — scans workspace/models/ and registers adapters.

Usage:
    from workspace.model_loader import scan_and_register
    registered = scan_and_register()  # returns list of registered model names
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "workspace" / "models"

# Ensure project src is importable
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def scan_and_register() -> List[str]:
    """Scan workspace/models/*.py, find BaseModelAdapter subclasses, register them.

    Returns list of newly registered model names.
    """
    from agent_market.freqai.model.base import BaseModelAdapter, ModelRegistry

    registered: List[str] = []
    if not MODELS_DIR.is_dir():
        return registered

    for py_file in sorted(MODELS_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"workspace_models_{py_file.stem}"
        if module_name in sys.modules:
            # Already loaded — check if it has registered adapters
            mod = sys.modules[module_name]
        else:
            try:
                spec = importlib.util.spec_from_file_location(module_name, str(py_file))
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
            except Exception as exc:
                print(f"[model_loader] failed to load {py_file.name}: {exc}")
                continue

        # Find and register any BaseModelAdapter subclasses
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModelAdapter)
                and obj is not BaseModelAdapter
                and hasattr(obj, "registry_name")
                and obj.registry_name != "base"
            ):
                name = obj.registry_name
                if name not in ModelRegistry._registry:
                    ModelRegistry.register(name, obj)
                    registered.append(name)
                    print(f"[model_loader] registered: {name} from {py_file.name}")

    return registered


def list_available_models() -> List[str]:
    """List all registered model names (built-in + workspace)."""
    from agent_market.freqai.model.base import ModelRegistry
    import agent_market.freqai.model  # noqa: F401 — trigger built-in registration

    scan_and_register()
    return sorted(ModelRegistry._registry.keys())


__all__ = ["scan_and_register", "list_available_models"]

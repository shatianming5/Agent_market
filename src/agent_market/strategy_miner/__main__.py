"""Allow running as: python -m agent_market.strategy_miner

Codex review R3-R4 (project-quality loop): this entrypoint previously
duplicated `scripts/strategy_miner.py`'s argparse + config resolver, but
with a silent ``MinerConfig.from_dict({})`` default fallback when no
``--config`` was passed. That bypassed the fail-close logic enforced by
the script entry. To prevent dual-entry drift, this module now **delegates
to the script entrypoint's main()** so config resolution is single-sourced.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Delegate to ``scripts/strategy_miner.py::main()`` so the
    ``--config`` / ``--resume`` / ``--allow-defaults`` fail-close logic is
    applied uniformly regardless of whether the user runs
    ``python scripts/strategy_miner.py`` or
    ``python -m agent_market.strategy_miner``.
    """
    # Locate the script entrypoint
    repo = Path(__file__).resolve().parents[3]
    script_dir = repo / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # The script imports as a module named ``strategy_miner`` (its filename),
    # which collides with our package name. Use importlib to load it
    # explicitly from the file path.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_strategy_miner_script_main", script_dir / "strategy_miner.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"strategy_miner __main__: failed to locate scripts/strategy_miner.py "
            f"under {script_dir}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(
            "scripts/strategy_miner.py has no main() — cannot delegate from "
            "module entrypoint."
        )
    rc = module.main()
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())

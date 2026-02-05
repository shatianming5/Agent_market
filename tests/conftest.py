from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _bootstrap_isolated_io_roots() -> None:
    """Isolate test runs from tracked evidence artifacts.

    Many tests run flow scripts that create artifacts under `artifacts/` and `user_data/`.
    This repo also tracks a subset of those for plan-grade evidence, so tests must not
    overwrite them. We redirect output roots into ignored subdirectories.
    """

    session = os.environ.get("AGENT_MARKET_PYTEST_SESSION") or uuid.uuid4().hex[:8]
    os.environ.setdefault("AGENT_MARKET_PYTEST_SESSION", session)

    os.environ.setdefault("AGENT_MARKET_ARTIFACTS_ROOT", f"artifacts/_pytest/{session}")
    os.environ.setdefault("AGENT_MARKET_USER_DATA_ROOT", f"user_data/_pytest/{session}")

    from agent_market import paths  # type: ignore

    artifacts_root = paths.artifacts_root()
    user_data_root = paths.user_data_root()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    user_data_root.mkdir(parents=True, exist_ok=True)

    # Copy minimal static inputs required by flow configs.
    for name in ("config_freqai_kucoin.json", "config_freqai.json"):
        src = (ROOT / "user_data" / name).resolve()
        dst = (user_data_root / name).resolve()
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if dst.exists():
            try:
                import json

                payload = json.loads(dst.read_text(encoding="utf-8-sig"))
                if isinstance(payload, dict):
                    payload["datadir"] = str((user_data_root / "data").resolve())
                    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    src_strategies = (ROOT / "user_data" / "strategies").resolve()
    dst_strategies = (user_data_root / "strategies").resolve()
    if src_strategies.exists() and not dst_strategies.exists():
        shutil.copytree(src_strategies, dst_strategies)


_bootstrap_isolated_io_roots()

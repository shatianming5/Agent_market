#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from agent_market import paths  # noqa: E402


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copyfile(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _prepare_isolated_user_data(user_root: Path) -> None:
    user_root.mkdir(parents=True, exist_ok=True)
    (user_root / "data").mkdir(parents=True, exist_ok=True)

    # Minimal inputs required by the closed-loop fixture config.
    _copyfile(paths.resolve_repo_path("user_data/config_freqai_kucoin.json"), user_root / "config_freqai_kucoin.json")
    _copytree(paths.resolve_repo_path("user_data/strategies"), user_root / "strategies")

    # Ensure datadir points at the isolated workspace (avoid writing into tracked `user_data/`).
    cfg_path = (user_root / "config_freqai_kucoin.json").resolve()
    cfg = _read_json(cfg_path)
    cfg["datadir"] = str((user_root / "data").resolve())
    _write_json(cfg_path, cfg)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 90-day closed-loop demo in an isolated workspace.")
    parser.add_argument(
        "--config",
        default="configs/agent_flow_closed_loop_demo_fixture.json",
        help="AgentFlow config (default: closed-loop fixture demo)",
    )
    parser.add_argument(
        "--steps",
        default="capture,lob_rebuild,micro_feature,factor_compile,factor_eval,ml,backtest,tca,report",
        help="Comma-separated steps (default: full closed loop)",
    )
    parser.add_argument(
        "--out-prefix",
        default="artifacts/_closed_loop_demo",
        help="Artifacts root prefix under repo (default: artifacts/_closed_loop_demo)",
    )
    args = parser.parse_args(argv)

    run_tag = f"{_utc_now_compact()}-{uuid.uuid4().hex[:8]}"
    artifacts_root = paths.resolve_repo_path(str(args.out_prefix)) / run_tag
    user_root = paths.resolve_repo_path("user_data/_closed_loop_demo") / run_tag

    _prepare_isolated_user_data(user_root)

    env = os.environ.copy()
    env["AGENT_MARKET_ARTIFACTS_ROOT"] = str(artifacts_root)
    env["AGENT_MARKET_USER_DATA_ROOT"] = str(user_root)
    env["PYTHONPATH"] = str(SRC)

    steps = [s.strip() for s in str(args.steps).replace(",", " ").split() if s.strip()]
    cmd = [
        sys.executable,
        str((ROOT / "scripts" / "agent_flow.py").resolve()),
        "--config",
        str(paths.resolve_repo_path(str(args.config))),
        "--steps",
        *steps,
    ]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)  # noqa: S603,S607

    # Print the generated run_id for convenience.
    latest_meta = (artifacts_root / "run_meta.json").resolve()
    if latest_meta.exists():
        try:
            meta = _read_json(latest_meta)
            rid = meta.get("run_id")
            print(f"[closed_loop_demo] run_id={rid}")
            print(f"[closed_loop_demo] artifacts_root={artifacts_root}")
            print(f"[closed_loop_demo] user_data_root={user_root}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


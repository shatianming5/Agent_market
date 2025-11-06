from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient


@dataclass
class _FakeJob:
    id: str
    cmd: List[str]
    cwd: Optional[Path]
    env: Optional[dict]
    logs: List[str] = field(default_factory=list)
    returncode: Optional[int] = None
    running: bool = True


class FakeJobManager:
    """A lightweight in-process JobManager stub for API testing."""

    def __init__(self) -> None:
        self._jobs: Dict[str, _FakeJob] = {}

    def start(self, cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = _FakeJob(id=job_id, cmd=list(cmd), cwd=cwd, env=env)
        self._jobs[job_id] = job

        def _run() -> None:
            job.logs.append("[fake] job started")
            job.logs.append("[fake] cmd: " + " ".join(cmd))
            time.sleep(0.05)
            job.logs.append("[fake] job finished")
            job.returncode = 0
            job.running = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return job_id

    def status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        return {
            "id": job.id,
            "cmd": job.cmd,
            "running": job.running,
            "returncode": job.returncode,
            "lines": len(job.logs),
        }

    def logs(self, job_id: str, offset: int = 0) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        total = len(job.logs)
        start = max(0, min(offset, total))
        chunk = job.logs[start:]
        return {
            "id": job.id,
            "offset": start,
            "next": start + len(chunk),
            "running": job.running,
            "returncode": job.returncode,
            "logs": chunk,
        }


def ensure_feature_file() -> Path:
    root = Path(__file__).resolve().parents[2]
    user_data = root / "user_data"
    user_data.mkdir(exist_ok=True)
    fpath = user_data / "freqai_features.json"
    if not fpath.exists():
        payload = {
            "timeframe": "1h",
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "features": [{"name": "feat_demo", "type": "sma_pct", "period": 5}],
        }
        fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fpath


def main() -> None:
    # Ensure repository root is importable before importing server
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Import after sys.path setup
    import server.main as m

    # Inject fake job manager so endpoints don't spawn real processes
    m.jobs = FakeJobManager()

    client = TestClient(m.app)
    root = Path(__file__).resolve().parents[2]

    print("[开始] 后端 API 冒烟/功能测试\n")

    # 1) /health
    print("[测试] GET /health —— 健康检查")
    r = client.get("/health")
    print("状态码:", r.status_code, "响应:", r.json())
    print()

    # 2) /run/expression (成功路径)
    print("[测试] POST /run/expression —— 启动表达式生成任务（成功路径，使用伪 JobManager）")
    feature_path = ensure_feature_file()
    expr_payload = {
        "config": "configs/config_freqai_multi.json",
        "feature_file": str(feature_path.relative_to(root)),
        "output": "user_data/freqai_expressions.json",
        "timeframe": "1h",
        "llm_model": "gpt-3.5-turbo",
        "llm_count": 2,
        "llm_loops": 1,
        "llm_timeout": 5,
        "feedback_top": 0,
    }
    r = client.post("/run/expression", json=expr_payload)
    print("状态码:", r.status_code, "响应键:", list(r.json().keys()))
    expr_job_id = r.json().get("job_id")
    print("返回 job_id:", expr_job_id)
    # 轮询一次日志
    time.sleep(0.1)
    rlog = client.get(f"/jobs/{expr_job_id}/logs")
    print("日志条数:", len(rlog.json().get("logs", [])))
    print()

    # 3) /run/expression（失败路径：缺少 feature_file）
    print("[测试] POST /run/expression —— 失败路径（缺少 feature_file）")
    bad_payload = dict(expr_payload)
    bad_payload["feature_file"] = "user_data/does_not_exist.json"
    r = client.post("/run/expression", json=bad_payload)
    print("状态码:", r.status_code, "响应:", r.json())
    print()

    # 4) /run/backtest（成功路径）
    print("[测试] POST /run/backtest —— 启动回测任务（成功路径，使用伪 JobManager）")
    bt_payload = {
        "config": "configs/config_freqai_multi.json",
        "strategy": "ExpressionLongStrategy",
        "strategy_path": "resources/user_data/strategies",
        "timerange": "20210101-20211231",
        "freqaimodel": "LightGBMRegressor",
        "export": False,
    }
    r = client.post("/run/backtest", json=bt_payload)
    print("状态码:", r.status_code, "响应键:", list(r.json().keys()))
    bt_job_id = r.json().get("job_id")
    print("返回 job_id:", bt_job_id)
    time.sleep(0.1)
    rstat = client.get(f"/jobs/{bt_job_id}/status")
    print("状态: running=", rstat.json().get("running"), "returncode=", rstat.json().get("returncode"))
    print()

    # 5) /run/backtest（失败路径：缺少 config）
    print("[测试] POST /run/backtest —— 失败路径（缺少 config 文件）")
    bad_bt = dict(bt_payload)
    bad_bt["config"] = "configs/not_found.json"
    r = client.post("/run/backtest", json=bad_bt)
    print("状态码:", r.status_code, "响应:", r.json())
    print()

    # 6) /flow/run（成功路径）
    print("[测试] POST /flow/run —— 启动 AgentFlow（成功路径，使用伪 JobManager）")
    flow_payload = {"config": "configs/agent_flow_example.json", "steps": "expression backtest"}
    r = client.post("/flow/run", json=flow_payload)
    print("状态码:", r.status_code, "响应键:", list(r.json().keys()))
    flow_job_id = r.json().get("job_id")
    print("返回 job_id:", flow_job_id)
    time.sleep(0.1)
    rlog2 = client.get(f"/jobs/{flow_job_id}/logs")
    print("日志条数:", len(rlog2.json().get("logs", [])))
    print()

    print("[完成] 后端 API 接口测试结束")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import logging
from fastapi import FastAPI, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from .job_manager import JobManager
from .config import load_settings, Settings
from .db import DB
from . import routes_agents, routes_orders

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

SETTINGS = load_settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title=SETTINGS.api_title, version=SETTINGS.api_version)
app.add_middleware(CORSMiddleware, allow_origins=SETTINGS.allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
jobs = JobManager(max_seconds=SETTINGS.job_max_seconds)

# Initialize database
DB_HANDLE = DB(SETTINGS.db_path)
DB_HANDLE.init()

# Dependency override providers for routers
def _get_db() -> DB:
    return DB_HANDLE

app.dependency_overrides[routes_agents.get_db] = _get_db
app.dependency_overrides[routes_orders.get_db] = _get_db

app.include_router(routes_agents.router)
app.include_router(routes_orders.router)


def _conda_prefix_args() -> list[str] | None:
    """Return ['conda','run','--no-capture-output','-n','freqtrade'] if conda env 'freqtrade' exists.

    Falls back to None if conda not available or env missing.
    """
    try:
        # Lazy import to avoid hard dependency
        import subprocess
        out = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if out.returncode == 0 and 'freqtrade' in out.stdout:
            return ['conda', 'run', '--no-capture-output', '-n', 'freqtrade']
    except Exception:
        pass
    return None


def _conda_env_python(env_name: str = 'freqtrade') -> Optional[str]:
    """Return absolute path to Python executable inside the given conda env, if available."""
    try:
        import subprocess, os
        out = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[0]
            if name != env_name:
                continue
            # Last column is the path
            prefix = parts[-1]
            if os.name == 'nt':
                cand = Path(prefix) / 'python.exe'
            else:
                cand = Path(prefix) / 'bin' / 'python'
            return str(cand) if cand.exists() else None
    except Exception:
        return None
    return None


class ExpressionReq(BaseModel):
    config: str = Field(..., description="Path to freqtrade JSON config")
    feature_file: str = Field(..., description="Path to feature json")
    output: str = Field("user_data/freqai_expressions.json")
    timeframe: str = Field("4h")
    llm_model: str = Field("gpt-3.5-turbo")
    llm_count: int = 20
    llm_loops: int = 1
    llm_timeout: float = 60
    llm_api_key: Optional[str] = None
    feedback: Optional[str] = None
    feedback_top: int = 0
    # 性能优化开关与可选阈值（不传则使用快速默认）
    fast: bool = True
    top: Optional[int] = None
    combo_top: Optional[int] = None
    stability_windows: Optional[int] = None
    stability_min_samples: Optional[int] = None
    force_fast: Optional[bool] = None


class BacktestReq(BaseModel):
    config: str = Field(...)
    strategy: str = Field("ExpressionLongStrategy")
    strategy_path: str = Field("freqtrade/user_data/strategies")
    timerange: str = Field("20210101-20211231")
    freqaimodel: str = Field("LightGBMRegressor")
    export: bool = True
    export_filename: str = Field("user_data/backtest_results/latest_trades_multi")
    fast: bool = True
    force_fast: Optional[bool] = None


class FlowReq(BaseModel):
    config: str = Field(..., description="Path to agent_flow JSON config")
    steps: Optional[str] = Field(None, description="Space separated steps e.g. 'feature expression backtest'")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/run/expression")
def run_expression(req: ExpressionReq = Body(...)):
    # Resolve and validate important paths
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    ff_path = Path(req.feature_file)
    if not ff_path.is_absolute():
        ff_path = (ROOT / ff_path).resolve()
    if not ff_path.exists():
        return {"status": "error", "code": "FEATURE_FILE_NOT_FOUND", "message": f"Feature file not found: {ff_path}"}
    conda_prefix = _conda_prefix_args()
    env_py = _conda_env_python('freqtrade')
    py = env_py or ('python' if conda_prefix else sys.executable)
    wrapper = ROOT / 'scripts' / 'expr_agent_wrapper.py'
    script_path = ROOT / 'freqtrade' / 'scripts' / 'freqai_expression_agent.py'
    if wrapper.exists():
        base = [py, str(wrapper)]
        feature_file_arg = str(ff_path)
    elif script_path.exists():
        base = [py, str(script_path)]
        feature_file_arg = str(ff_path)
    else:
        # fallback to module invocation
        base = [py, '-m', 'freqtrade.scripts.freqai_expression_agent']
        feature_file_arg = req.feature_file
    cmd = (conda_prefix or []) + base
    cmd += [
        '--config', str(cfg_path),
        '--feature-file', feature_file_arg,
        '--output', req.output,
        '--timeframe', req.timeframe,
        '--llm-model', req.llm_model,
        '--llm-count', str(req.llm_count),
        '--llm-loops', str(req.llm_loops),
        '--llm-timeout', str(req.llm_timeout),
        '--feedback-top', str(req.feedback_top),
    ]
    # 速度优先：限制候选规模，关闭组合特征，降低稳定性窗口
    fast_mode = req.fast or (req.force_fast if req.force_fast is not None else SETTINGS.force_fast)
    if fast_mode:
        cmd += [
            '--top', str(req.top or 40),
            '--combo-top', str(req.combo_top or 0),
            '--stability-windows', str(req.stability_windows or 1),
            '--stability-min-samples', str(req.stability_min_samples or 50),
            '--backtest-weight', '0.0',
        ]
    if req.feedback:
        cmd += ['--feedback', req.feedback]

    env = os.environ.copy()
    # 仅确保 src/ 可导入（避免覆盖 conda 内已安装的 freqtrade 包）
    env['PYTHONPATH'] = os.pathsep.join([str(SRC), str(ROOT / 'freqtrade'), env.get('PYTHONPATH', '')])
    # LLM credentials from request or environment
    if req.llm_api_key:
        env['LLM_API_KEY'] = req.llm_api_key
    if os.environ.get('LLM_BASE_URL'):
        env['LLM_BASE_URL'] = os.environ['LLM_BASE_URL']
    if os.environ.get('LLM_MODEL'):
        env['LLM_MODEL'] = os.environ['LLM_MODEL']
    # If no API key is present anywhere, avoid LLM usage to prevent failures
    if not env.get('LLM_API_KEY') and 'LLM_API_KEY' not in os.environ:
        cmd += ['--no-llm', '--gp-enabled']
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.post("/run/backtest")
def run_backtest(req: BacktestReq = Body(...)):
    # Resolve and validate config and strategy path
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    spath = Path(req.strategy_path) if req.strategy_path else None
    if spath and not spath.is_absolute():
        spath = (ROOT / spath).resolve()
    if spath and not spath.exists():
        return {"status": "error", "code": "STRATEGY_PATH_NOT_FOUND", "message": f"Strategy path not found: {spath}"}

    env_py = _conda_env_python('freqtrade')
    conda_prefix = _conda_prefix_args() if not env_py else None
    py = env_py or ('python' if conda_prefix else sys.executable)
    binary = 'freqtrade'
    from shutil import which
    # Prefer direct env python if available to avoid "conda run" stdio issues
    if env_py:
        base = [py, '-m', 'freqtrade', 'backtesting']
    else:
        # If no direct env python, try console script, else fallback to -m
        use_bin = which(binary) is not None and conda_prefix is None
        if use_bin:
            base = [binary, 'backtesting']
        else:
            base = [py, '-m', 'freqtrade', 'backtesting']
    # 短时间范围优先（除非显式指定 fast=False）
    fast_mode = req.fast or (req.force_fast if req.force_fast is not None else SETTINGS.force_fast)
    timerange_value = '20210101-20210131' if fast_mode else req.timerange
    cmd = (conda_prefix or []) + base + [
        '--config', str(cfg_path),
        '--strategy', req.strategy,
        '--strategy-path', str(spath) if spath else req.strategy_path,
        '--timerange', timerange_value,
        '--freqaimodel', req.freqaimodel,
    ]
    if req.export:
        cmd += ['--export', 'trades', '--export-filename', req.export_filename]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(SRC), str(ROOT / 'freqtrade'), env.get('PYTHONPATH', '')])
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.post("/flow/run")
def run_flow(req: FlowReq = Body(...)):
    py = sys.executable
    script = str(ROOT / 'scripts' / 'agent_flow.py')
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    cmd = [py, script, '--config', str(cfg_path)]
    if req.steps:
        parts = [p for p in req.steps.split(' ') if p]
        if parts:
            cmd += ['--steps'] + parts
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(SRC), str(ROOT / 'freqtrade'), env.get('PYTHONPATH', '')])
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"job_id": job_id, "cmd": cmd}


@app.get('/jobs/{job_id}/status')
def job_status(job_id: str):
    return jobs.status(job_id)


@app.get('/jobs')
def job_list():
    return jobs.list()


@app.get('/jobs/{job_id}/logs')
def job_logs(job_id: str, offset: int = 0, limit: int = 200, structured: bool = False):
    res = jobs.logs(job_id, offset)
    try:
        lim = max(1, min(int(limit), 1000))
    except Exception:
        lim = 200
    logs = res.get('logs', [])
    truncated = logs[:lim]
    res['logs'] = truncated
    if structured:
        start = res.get('offset', 0)
        res['entries'] = [{"line": start + i, "text": v} for i, v in enumerate(truncated)]
    return res


@app.post('/jobs/{job_id}/terminate')
def job_terminate(job_id: str):
    return jobs.terminate(job_id)


@app.get('/jobs/{job_id}/stream')
def job_stream(job_id: str, offset: int = 0):
    import time

    def gen():
        off = max(0, int(offset))
        keepalive = 0
        while True:
            data = jobs.logs(job_id, off)
            logs = data.get('logs', [])
            nxt = data.get('next', off)
            for line in logs:
                yield f"data: {line}\n\n"
            off = nxt
            if not data.get('running'):
                yield "event: end\n"
                yield "data: finished\n\n"
                break
            keepalive += 1
            if keepalive % 10 == 0:
                yield ": keep-alive\n\n"
            time.sleep(0.5)

    return StreamingResponse(gen(), media_type='text/event-stream')


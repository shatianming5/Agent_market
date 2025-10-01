from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .job_manager import JobManager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

app = FastAPI(title="Agent Market Server", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
jobs = JobManager()


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


class BacktestReq(BaseModel):
    config: str = Field(...)
    strategy: str = Field("ExpressionLongStrategy")
    strategy_path: str = Field("freqtrade/user_data/strategies")
    timerange: str = Field("20210101-20211231")
    freqaimodel: str = Field("LightGBMRegressor")
    export: bool = True
    export_filename: str = Field("user_data/backtest_results/latest_trades_multi")


class FlowReq(BaseModel):
    config: str = Field(..., description="Path to agent_flow JSON config")
    steps: Optional[str] = Field(None, description="Space separated steps e.g. 'feature expression backtest'")


@app.get("/health")
def health():
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
    py = sys.executable
    wrapper = ROOT / 'scripts' / 'expr_agent_wrapper.py'
    script_path = ROOT / 'freqtrade' / 'scripts' / 'freqai_expression_agent.py'
    if wrapper.exists():
        cmd = [py, str(wrapper)]
        feature_file_arg = str(ff_path)
    elif script_path.exists():
        cmd = [py, str(script_path)]
        feature_file_arg = req.feature_file
    else:
        # fallback to module invocation
        cmd = [py, '-m', 'freqtrade.scripts.freqai_expression_agent']
        feature_file_arg = req.feature_file
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
    if req.feedback:
        cmd += ['--feedback', req.feedback]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    # LLM credentials from request or environment
    if req.llm_api_key:
        env['LLM_API_KEY'] = req.llm_api_key
    if os.environ.get('LLM_BASE_URL'):
        env['LLM_BASE_URL'] = os.environ['LLM_BASE_URL']
    if os.environ.get('LLM_MODEL'):
        env['LLM_MODEL'] = os.environ['LLM_MODEL']
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

    py = sys.executable
    binary = 'freqtrade'
    if os.name == 'nt':
        # check availability
        from shutil import which
        has_bin = which(binary) is not None
    else:
        from shutil import which
        has_bin = which(binary) is not None

    if has_bin:
        cmd = [
            binary, 'backtesting',
            '--config', str(cfg_path),
            '--strategy', req.strategy,
            '--strategy-path', str(spath) if spath else req.strategy_path,
            '--timerange', req.timerange,
            '--freqaimodel', req.freqaimodel,
        ]
    else:
        cmd = [
            py, '-m', 'freqtrade', 'backtesting',
            '--config', str(cfg_path),
            '--strategy', req.strategy,
            '--strategy-path', str(spath) if spath else req.strategy_path,
            '--timerange', req.timerange,
            '--freqaimodel', req.freqaimodel,
        ]
    if req.export:
        cmd += ['--export', 'trades', '--export-filename', req.export_filename]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
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
    env['PYTHONPATH'] = str(SRC)
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"job_id": job_id, "cmd": cmd}


@app.get('/jobs/{job_id}/status')
def job_status(job_id: str):
    return jobs.status(job_id)


@app.get('/jobs/{job_id}/logs')
def job_logs(job_id: str, offset: int = 0):
    return jobs.logs(job_id, offset)


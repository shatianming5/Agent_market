from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

from .job_manager import JobManager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

app = FastAPI(title="Agent Market Server", version="0.1.0")
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
    py = sys.executable
    script = str(ROOT / 'freqtrade' / 'scripts' / 'freqai_expression_agent.py')
    cmd = [
        py,
        script,
        '--config', req.config,
        '--feature-file', req.feature_file,
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
    if req.llm_api_key:
        env['LLM_API_KEY'] = req.llm_api_key
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"job_id": job_id, "cmd": cmd}


@app.post("/run/backtest")
def run_backtest(req: BacktestReq = Body(...)):
    cmd = [
        'freqtrade', 'backtesting',
        '--config', req.config,
        '--strategy', req.strategy,
        '--strategy-path', req.strategy_path,
        '--timerange', req.timerange,
        '--freqaimodel', req.freqaimodel,
    ]
    if req.export:
        cmd += ['--export', 'trades', '--export-filename', req.export_filename]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"job_id": job_id, "cmd": cmd}


@app.post("/flow/run")
def run_flow(req: FlowReq = Body(...)):
    py = sys.executable
    script = str(ROOT / 'scripts' / 'agent_flow.py')
    cmd = [py, script, '--config', req.config]
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


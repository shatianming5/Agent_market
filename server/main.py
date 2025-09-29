from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .job_manager import JobManager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

app = FastAPI(title="Agent Market Server", version="0.1.0")
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


class FeatureReq(BaseModel):
    config: str = Field(...)
    output: str = Field("user_data/freqai_features.json")
    timeframe: str = Field("4h")
    pairs: Optional[str] = Field(None, description="Comma or space separated pairs, e.g. 'BTC/USDT ETH/USDT'")


class HyperoptReq(BaseModel):
    config: str = Field(...)  # freqtrade config
    strategy: str = Field("ExpressionLongStrategy")
    strategy_path: str = Field("freqtrade/user_data/strategies")
    timerange: str = Field("20210101-20210430")
    spaces: str = Field("buy sell protection")
    hyperopt_loss: str = Field("SharpeHyperOptLoss")
    epochs: int = Field(20)
    freqaimodel: Optional[str] = Field("LightGBMRegressor")
    job_workers: int = Field(-1)


@app.get('/features/top')
def features_top(file: str = 'user_data/freqai_features.json', limit: int = 20):
    """Return top-N features by score/correlation/mutual_info."""
    path = Path(file)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        return {"error": f"Feature file not found: {path}"}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {"error": f"Failed to parse: {exc}"}
    feats = payload.get('features') or []
    rows = []
    for it in feats:
        name = it.get('name')
        if not name:
            continue
        score = it.get('score')
        if score is None:
            score = it.get('correlation')
        if score is None:
            score = it.get('mutual_info')
        val = None
        try:
            val = float(score) if score is not None else None
        except Exception:
            val = None
        rows.append({
            'name': name,
            'type': it.get('type'),
            'period': it.get('period'),
            'score': val,
            'correlation': it.get('correlation'),
            'mutual_info': it.get('mutual_info'),
            'description': it.get('description'),
        })
    rows.sort(key=lambda r: abs(r['score']) if isinstance(r['score'], (int,float)) else 0.0, reverse=True)
    return {'file': str(path), 'total': len(rows), 'items': rows[:max(1, int(limit))]}


@app.get('/results/list')
def results_list(results_dir: str = 'user_data/backtest_results', limit: int = 20):
    rd = Path(results_dir)
    if not rd.is_absolute():
        rd = (ROOT / rd).resolve()
    if not rd.exists():
        return {"error": f"Results dir not found: {rd}"}
    items = []
    for p in rd.glob('backtest-result-*.zip'):
        items.append({'name': p.name, 'mtime': p.stat().st_mtime, 'size': p.stat().st_size})
    items.sort(key=lambda x: x['mtime'], reverse=True)
    return {'dir': str(rd), 'items': items[:max(1, int(limit))]}


@app.get('/results/summary')
def results_summary(name: str, results_dir: str = 'user_data/backtest_results'):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig  # type: ignore
    rd = Path(results_dir)
    zp = rd / name
    if not zp.exists():
        return {"error": f"Not found: {zp}"}
    flow = AgentFlow(AgentFlowConfig())
    summary = flow._build_backtest_summary(zp)
    return summary


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run/expression")
def run_expression(req: ExpressionReq = Body(...)):
    py = sys.executable
    script = str(ROOT / 'freqtrade' / 'scripts' / 'freqai_expression_agent.py')
    feature_file = req.feature_file
    if feature_file and not Path(feature_file).is_absolute() and feature_file.replace('\\','/').startswith('user_data/'):
        feature_file = '../' + feature_file.replace('\\','/')

    cmd = [
        py,
        script,
        '--config', req.config,
        '--feature-file', feature_file,
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


@app.post("/run/hyperopt")
def run_hyperopt(req: HyperoptReq = Body(...)):
    cmd = [
        'freqtrade', 'hyperopt',
        '--config', req.config,
        '--strategy', req.strategy,
        '--strategy-path', req.strategy_path,
        '--timerange', req.timerange,
        '--hyperopt-loss', req.hyperopt_loss,
        '--epochs', str(req.epochs),
        '--job-workers', str(req.job_workers),
    ]
    if req.spaces:
        cmd += ['--spaces'] + req.spaces.split()
    if req.freqaimodel:
        cmd += ['--freqaimodel', req.freqaimodel]
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


@app.post("/run/feature")
def run_feature(req: FeatureReq = Body(...)):
    py = sys.executable
    script = str(ROOT / 'freqtrade' / 'scripts' / 'freqai_feature_agent.py')
    cmd = [
        py,
        script,
        '--config', req.config,
        '--output', req.output,
        '--timeframe', req.timeframe,
    ]
    if req.pairs:
        # split by comma or whitespace
        parts = [p.strip() for p in req.pairs.replace(',', ' ').split() if p.strip()]
        if parts:
            cmd += ['--pairs'] + parts
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


# Results summary
def _find_latest_zip(results_dir: Path) -> Optional[Path]:
    results_dir = results_dir.resolve()
    last = results_dir / '.last_result.json'
    if last.exists():
        try:
            latest = json.loads(last.read_text(encoding='utf-8')).get('latest_backtest')  # type: ignore[name-defined]
            if latest:
                p = results_dir / latest
                if p.exists():
                    return p
        except Exception:
            pass
    zips = sorted(results_dir.glob('backtest-result-*.zip'), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


@app.get('/results/latest-summary')
def latest_summary(results_dir: str = 'user_data/backtest_results'):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig  # lazy import
    rd = Path(results_dir)
    zip_path = _find_latest_zip(rd)
    if not zip_path:
        return {"error": f"No backtest archives found in {rd}"}
    flow = AgentFlow(AgentFlowConfig())
    summary = flow._build_backtest_summary(zip_path)
    # try to enrich with trades details if available in zip
    try:
        import zipfile  # noqa: PLC0415
        with zipfile.ZipFile(zip_path) as zf:
            trade_members = [name for name in zf.namelist() if name.endswith('.json') and name.startswith('trades-')]
            if trade_members:
                import json as _json  # noqa: N816
                trades = _json.loads(zf.read(trade_members[0]))
                summary['trades'] = trades
    except Exception:
        pass
    return summary


@app.post('/results/prepare-feedback')
def prepare_feedback(results_dir: str = 'user_data/backtest_results', out: str = 'user_data/llm_feedback/latest_backtest_summary.json'):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig  # type: ignore
    rd = Path(results_dir)
    zip_path = _find_latest_zip(rd)
    if not zip_path:
        return {"error": f"No backtest archives found in {rd}"}
    flow = AgentFlow(AgentFlowConfig())
    summary = flow._build_backtest_summary(zip_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return {"feedback_path": str(out_path)}

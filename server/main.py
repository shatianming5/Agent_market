from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, Body, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import shutil
import re
import time
import asyncio
import json as _jsonmod
from starlette.responses import StreamingResponse

from .job_manager import JobManager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'


def _load_dotenv_into_environ(env_path: Path) -> None:
    """Lightweight .env loader without extra deps.
    Supports KEY=VALUE lines, ignores comments and empty lines.
    """
    try:
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # populate os.environ if missing or empty
            if not os.environ.get(k):
                os.environ[k] = v
    except Exception:
        # best-effort; don't crash server startup
        pass


# Load .env from project root into process env
_load_dotenv_into_environ(ROOT / '.env')

app = FastAPI(title="Agent Market Server", version="0.2.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
jobs = JobManager()

# Serve static web UI (optional): http://host:8000/web/index.html
try:
    app.mount("/web", StaticFiles(directory=str(ROOT / 'web')), name="web")
except Exception:
    # Mounting static is best-effort; ignore if missing
    pass


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
    # Accept both space-separated string and list of steps from web client
    steps: Optional[object] = Field(None, description="Either space separated string or list: feature expression ml rl backtest")


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


class RLTrainReq(BaseModel):
    config: str = Field(..., description="Path to RL training JSON config (train_ppo.json)")


class TrainReq(BaseModel):
    config: Optional[str] = Field(None, description="Path to ML training JSON config (train_*.json)")
    config_obj: Optional[dict] = Field(None, description="Inline ML training config (JSON object)")


@app.get('/features/top')
def features_top(file: str = 'user_data/freqai_features.json', limit: int = 20):
    """Return top-N features by score/correlation/mutual_info."""
    path = Path(file)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        return _error("FEATURE_FILE_NOT_FOUND", f"Feature file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return _error("PARSE_ERROR", f"Failed to parse: {exc}")
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
        alt = (ROOT / 'freqtrade' / 'user_data' / 'backtest_results').resolve()
        if alt.exists():
            rd = alt
        else:
            return _error("RESULTS_DIR_NOT_FOUND", f"Results dir not found: {rd}")
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
        alt = (ROOT / 'freqtrade' / 'user_data' / 'backtest_results' / name).resolve()
        if alt.exists():
            zp = alt
        else:
            return _error("NOT_FOUND", f"Not found: {zp}")
    flow = AgentFlow(AgentFlowConfig())
    summary = flow._build_backtest_summary(zp)
    # enrich with trades if present
    try:
        import zipfile  # noqa: PLC0415
        with zipfile.ZipFile(zp) as zf:
            trade_members = [nm for nm in zf.namelist() if nm.endswith('.json') and nm.startswith('trades-')]
            if trade_members:
                import json as _json  # noqa: N816
                trades = _json.loads(zf.read(trade_members[0]))
                summary['trades'] = trades
    except Exception:
        pass
    return summary


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "Agent Market API",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/run/expression")
def run_expression(req: ExpressionReq = Body(...)):
    # Validate config and feature file paths for clearer errors
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
    if not _validate_timeframe(req.timeframe):
        return {"status": "error", "code": "INVALID_TIMEFRAME", "message": f"Invalid timeframe: {req.timeframe}"}
    try:
        if req.llm_count is not None and int(req.llm_count) <= 0:
            return {"status": "error", "code": "INVALID_LLM_COUNT", "message": f"llm_count must be > 0, got {req.llm_count}"}
    except Exception:
        return {"status": "error", "code": "INVALID_LLM_COUNT", "message": f"llm_count must be an integer, got {req.llm_count}"}
    if not req.llm_model:
        return {"status": "error", "code": "INVALID_LLM_MODEL", "message": "llm_model must be provided"}
    py = sys.executable
    # Prefer wrapper to add retry/backoff/fallback
    wrapper = ROOT / 'scripts' / 'expr_agent_wrapper.py'
    script_path = ROOT / 'freqtrade' / 'scripts' / 'freqai_expression_agent.py'
    raw_feature_file = req.feature_file
    if wrapper.exists():
        cmd = [py, str(wrapper)]
        # Use absolute path to avoid freqtrade script resolving relative to config dir
        feature_file_arg = str(ff_path)
    elif script_path.exists():
        cmd = [py, str(script_path)]
        feature_file_arg = (
            ('../' + raw_feature_file.replace('\\','/'))
            if raw_feature_file and not Path(raw_feature_file).is_absolute() and raw_feature_file.replace('\\','/').startswith('user_data/')
            else raw_feature_file
        )
    else:
        # Fall back to module invocation when freqtrade is installed in environment
        cmd = [py, '-m', 'freqtrade.scripts.freqai_expression_agent']
        feature_file_arg = raw_feature_file
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
    # Prefer API key from request; fallback to .env / process env
    llm_key = req.llm_api_key or os.environ.get('LLM_API_KEY')
    if llm_key:
        env['LLM_API_KEY'] = llm_key
    # Also propagate base url / default model if present in env
    if os.environ.get('LLM_BASE_URL'):
        env['LLM_BASE_URL'] = os.environ['LLM_BASE_URL']
    if os.environ.get('LLM_MODEL'):
        env['LLM_MODEL'] = os.environ['LLM_MODEL']
    job_id = jobs.start(cmd, cwd=ROOT, env=env, timeout_sec=900, kind='expression', meta={"timeframe": req.timeframe})
    return {"status": "started", "job_id": job_id, "kind": "expression", "cmd": cmd}


@app.post("/run/backtest")
def run_backtest(req: BacktestReq = Body(...)):
    # Resolve and validate important paths for clearer error reporting
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return _error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")
    spath: Optional[Path] = None
    if req.strategy_path:
        spath = Path(req.strategy_path)
        if not spath.is_absolute():
            spath = (ROOT / spath).resolve()
        if not spath.exists():
            return {"status": "error", "code": "STRATEGY_PATH_NOT_FOUND", "message": f"Strategy path not found: {spath}"}
    py = sys.executable
    binary = 'freqtrade'
    if shutil.which(binary):
        cmd = [
            binary, 'backtesting',
            '--config', str(cfg_path),
            '--strategy', req.strategy,
            '--strategy-path', str(spath) if spath else req.strategy_path,
            '--timerange', req.timerange,
            '--freqaimodel', req.freqaimodel,
        ]
    else:
        # Fallback to module invocation when console script is unavailable
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
    job_id = jobs.start(cmd, cwd=ROOT, env=env, timeout_sec=7200, kind='backtest', meta={"timerange": req.timerange})
    return {"status": "started", "job_id": job_id, "kind": "backtest", "cmd": cmd}


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
    job_id = jobs.start(cmd, cwd=ROOT, env=env, timeout_sec=1800, kind='hyperopt', meta={"timerange": req.timerange})
    return {"status": "started", "job_id": job_id, "kind": "hyperopt", "cmd": cmd}


@app.post("/run/rl_train")
def run_rl_train(req: RLTrainReq = Body(...)):
    py = sys.executable
    script = str(ROOT / 'scripts' / 'train_rl.py')
    cmd = [py, script, '--config', req.config]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    job_id = jobs.start(cmd, cwd=ROOT, env=env, timeout_sec=7200, kind='rl_train')
    return {"status": "started", "job_id": job_id, "kind": "rl_train", "cmd": cmd}


@app.post("/run/train")
def run_train(req: TrainReq = Body(...)):
    py = sys.executable
    script = str(ROOT / 'scripts' / 'train_pipeline.py')
    cfg_path: Optional[Path] = None
    if req.config:
        cfg_path = (ROOT / req.config) if not Path(req.config).is_absolute() else Path(req.config)
        if not cfg_path.exists():
            return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    elif req.config_obj:
        tmp_dir = ROOT / 'user_data' / 'tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # generate unique filename
        from datetime import datetime  # noqa: PLC0415
        # validate inline config structure
        cfg = req.config_obj if isinstance(req.config_obj, dict) else None
        if not cfg:
            return {"status": "error", "code": "INVALID_BODY", "message": "config_obj must be a JSON object"}
        required_top = {"data", "model", "training", "output"}
        missing = [k for k in required_top if k not in cfg]
        if missing:
            return {"status": "error", "code": "MISSING_KEYS", "message": f"config_obj missing keys: {missing}"}
        data = cfg.get('data') or {}
        model = cfg.get('model') or {}
        training = cfg.get('training') or {}
        output = cfg.get('output') or {}
        # data checks
        for key in ("feature_file", "data_dir", "exchange", "timeframe"):
            if key not in data:
                return {"status": "error", "code": "MISSING_DATA_KEY", "message": f"data.{key} required"}
        if not _validate_timeframe(str(data.get('timeframe'))):
            return {"status": "error", "code": "INVALID_TIMEFRAME", "message": f"Invalid data.timeframe: {data.get('timeframe')}"}
        # pairs optional list
        pairs = data.get('pairs')
        if pairs is not None:
            if not isinstance(pairs, list) or not all(isinstance(p, str) for p in pairs):
                return {"status": "error", "code": "INVALID_PAIRS", "message": "data.pairs must be a list of strings"}
        # feature_file existence (best-effort)
        try:
            ff = Path(data.get('feature_file'))
            if not ff.is_absolute():
                ff = (ROOT / ff).resolve()
            if not ff.exists():
                return {"status": "error", "code": "FEATURE_FILE_NOT_FOUND", "message": f"Feature file not found: {ff}"}
        except Exception:
            return {"status": "error", "code": "FEATURE_FILE_INVALID", "message": f"Invalid feature_file: {data.get('feature_file')}"}
        # model checks
        if not model.get('name'):
            return {"status": "error", "code": "MODEL_NAME_REQUIRED", "message": "model.name required"}
        # training checks
        vr = float(training.get('validation_ratio', 0.2))
        if not (0.0 <= vr <= 0.9):
            return {"status": "error", "code": "INVALID_VALIDATION_RATIO", "message": f"training.validation_ratio out of range: {vr}"}
        # output checks
        out_dir = output.get('model_dir') or 'artifacts/models/auto'
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            return {"status": "error", "code": "MODEL_DIR_INVALID", "message": f"Cannot create output.model_dir: {out_dir}"}
        cfg_path = tmp_dir / f"train_inline_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        return {"status": "error", "code": "MISSING_CONFIG", "message": "Either 'config' (path) or 'config_obj' (inline JSON) must be provided"}
    cmd = [py, script, '--config', str(cfg_path)]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    job_id = jobs.start(cmd, cwd=ROOT, env=env, kind='train')
    return {"status": "started", "job_id": job_id, "kind": "train", "cmd": cmd}


@app.get('/results/latest-training')
def results_latest_training(models_dir: str = 'artifacts/models'):
    base = Path(models_dir)
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    if not base.exists():
        return _error("MODELS_DIR_NOT_FOUND", f"Models dir not found: {base}")
    candidates = list(base.rglob('training_summary.json'))
    if not candidates:
        return {"error": f"No training_summary.json under {base}"}
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding='utf-8'))
    except Exception as exc:  # pragma: no cover
        return _error("PARSE_ERROR", f"Failed to parse {latest}: {exc}")
    payload['summary_path'] = str(latest)
    return payload


@app.post("/flow/run")
def run_flow(req: FlowReq = Body(...)):
    py = sys.executable
    script = str(ROOT / 'scripts' / 'agent_flow.py')
    cmd = [py, script, '--config', req.config]
    if req.steps:
        parts: list[str] = []
        if isinstance(req.steps, str):
            parts = [p for p in req.steps.split(' ') if p]
        elif isinstance(req.steps, list):
            parts = [str(p) for p in req.steps if p]
        if parts:
            cmd += ['--steps'] + parts
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    job_id = jobs.start(cmd, cwd=ROOT, env=env, kind='flow')
    return {"status": "started", "job_id": job_id, "kind": "flow", "cmd": cmd}


@app.post("/run/feature")
def run_feature(req: FeatureReq = Body(...)):
    py = sys.executable
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    if not _validate_timeframe(req.timeframe):
        return {"status": "error", "code": "INVALID_TIMEFRAME", "message": f"Invalid timeframe: {req.timeframe}"}
    script_path = ROOT / 'freqtrade' / 'scripts' / 'freqai_feature_agent.py'
    if script_path.exists():
        cmd = [py, str(script_path)]
    else:
        # Fall back to module invocation when freqtrade is installed in environment
        cmd = [py, '-m', 'freqtrade.scripts.freqai_feature_agent']
    cmd += ['--config', str(cfg_path), '--output', req.output, '--timeframe', req.timeframe]
    if req.pairs:
        # split by comma or whitespace
        ok_pairs, parsed = _validate_pairs_string(req.pairs)
        if not ok_pairs:
            return {"status": "error", "code": "INVALID_PAIRS", "message": f"Invalid pairs string: {req.pairs}"}
        parts = parsed
        if parts:
            cmd += ['--pairs'] + parts
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    job_id = jobs.start(cmd, cwd=ROOT, env=env, kind='feature', meta={"timeframe": req.timeframe})
    return {"status": "started", "job_id": job_id, "kind": "feature", "cmd": cmd}


@app.get('/jobs/{job_id}/status')
def job_status(job_id: str):
    res = jobs.status(job_id)
    if isinstance(res, dict) and res.get('error'):
        return _error('JOB_NOT_FOUND', str(res.get('error')))
    return res


@app.get('/jobs/{job_id}/logs')
def job_logs(job_id: str, offset: int = 0):
    res = jobs.logs(job_id, offset)
    if isinstance(res, dict) and res.get('error'):
        return _error('JOB_NOT_FOUND', str(res.get('error')))
    return res


@app.post('/jobs/{job_id}/cancel')
def job_cancel(job_id: str):
    res = jobs.cancel(job_id)
    if isinstance(res, dict) and res.get('error'):
        return _error('JOB_NOT_FOUND', str(res.get('error')))
    return res


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
        alt = (ROOT / 'freqtrade' / 'user_data' / 'backtest_results').resolve()
        zip_path = _find_latest_zip(alt)
        if not zip_path:
            return _error("NO_ARCHIVES", f"No backtest archives found in {rd}")
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


@app.get("/index")
def index():
    return {
        "message": "Agent Market API",
        "docs": "/docs",
        "health": "/health"
    }

@app.post('/results/prepare-feedback')
def prepare_feedback(results_dir: str = 'user_data/backtest_results', out: str = 'user_data/llm_feedback/latest_backtest_summary.json'):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig  # type: ignore
    rd = Path(results_dir)
    zip_path = _find_latest_zip(rd)
    if not zip_path:
        alt = (ROOT / 'freqtrade' / 'user_data' / 'backtest_results').resolve()
        zip_path = _find_latest_zip(alt)
        if not zip_path:
            return _error("NO_ARCHIVES", f"No backtest archives found in {rd}")
    flow = AgentFlow(AgentFlowConfig())
    summary = flow._build_backtest_summary(zip_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return {"feedback_path": str(out_path)}


@app.get('/results/gallery')
def results_gallery(results_dir: str = 'user_data/backtest_results', limit: int = 20):
    rd = Path(results_dir)
    if not rd.is_absolute():
        rd = (ROOT / rd).resolve()
    if not rd.exists():
        alt = (ROOT / 'freqtrade' / 'user_data' / 'backtest_results').resolve()
        if alt.exists():
            rd = alt
        else:
            return _error("RESULTS_DIR_NOT_FOUND", f"Results dir not found: {rd}")
    zips = sorted(rd.glob('backtest-result-*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig  # type: ignore
    flow = AgentFlow(AgentFlowConfig())
    for zp in zips[:max(1, int(limit))]:
        try:
            summary = flow._build_backtest_summary(zp)
            summary['name'] = zp.name
            summary['mtime'] = zp.stat().st_mtime
            items.append(summary)
        except Exception:
            continue
    return {"dir": str(rd), "items": items}


@app.get('/results/aggregate')
def results_aggregate(names: str, results_dir: str = 'user_data/backtest_results'):
    rd = Path(results_dir)
    if not rd.is_absolute():
        rd = (ROOT / rd).resolve()
    if not rd.exists():
        alt = (ROOT / 'freqtrade' / 'user_data' / 'backtest_results').resolve()
        if alt.exists():
            rd = alt
        else:
            return _error("RESULTS_DIR_NOT_FOUND", f"Results dir not found: {rd}")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agent_market.agent_flow import AgentFlow, AgentFlowConfig  # type: ignore
    flow = AgentFlow(AgentFlowConfig())
    files = [rd / n for n in names.split(',') if n.strip()]
    metrics = []
    for f in files:
        if not f.exists():
            continue
        try:
            s = flow._build_backtest_summary(f)
            metrics.append({
                'name': f.name,
                'profit_total_pct': s.get('profit_total_pct'),
                'max_drawdown_abs': s.get('max_drawdown_abs'),
                'trades': s.get('trades'),
            })
        except Exception:
            continue
    if not metrics:
        return _error("NO_VALID_INPUTS", "no valid inputs")
    # simple robust score: mean_profit / (std_profit + 1e-9)
    import numpy as _np  # noqa: N816
    profs = _np.array([m.get('profit_total_pct') or 0.0 for m in metrics], dtype=float)
    mean = float(_np.mean(profs))
    std = float(_np.std(profs))
    score = mean / (std + 1e-9)
    return {"items": metrics, "mean_profit": mean, "std_profit": std, "robust_score": score}


# ------------------------------- Settings API -------------------------------

@app.get('/settings')
def get_settings():
    return _load_settings()


@app.post('/settings')
def post_settings(payload: dict = Body(...)):
    if not isinstance(payload, dict):
        return {"status": "error", "code": "INVALID_BODY", "message": "settings body must be a JSON object"}
    allowed = {"llm_base_url", "llm_model", "default_timeframe", "note"}
    updates = {k: v for k, v in payload.items() if k in allowed}
    if not updates:
        return {"status": "error", "code": "NO_ALLOWED_KEYS", "message": f"allowed keys: {sorted(allowed)}"}
    cur = _load_settings()
    cur.update(updates)
    # basic validation
    if cur.get('default_timeframe') and not _validate_timeframe(str(cur['default_timeframe'])):
        return {"status": "error", "code": "INVALID_TIMEFRAME", "message": f"Invalid default_timeframe: {cur['default_timeframe']}"}
    _save_settings(cur)
    return {"status": "ok", "settings": cur}
# -------------------------- Utilities & Settings --------------------------

_TIMEFRAME_RE = re.compile(r"^\s*\d+(ms|s|m|h|d|w)\s*$", re.IGNORECASE)


def _error(code: str, message: str, **extra) -> dict:
    payload = {"status": "error", "code": code, "message": message}
    if extra:
        payload.update(extra)
    return payload


def _validate_timeframe(tf: str) -> bool:
    return bool(tf and _TIMEFRAME_RE.match(str(tf)))


def _validate_pairs_string(pairs: Optional[str]) -> Tuple[bool, list[str]]:
    if not pairs:
        return True, []
    raw = pairs.replace(',', ' ').split()
    ok = []
    for token in raw:
        if re.match(r"^[A-Za-z0-9]+/[A-Za-z0-9]+$", token):
            ok.append(token)
        else:
            return False, []
    return True, ok


SETTINGS_PATH = ROOT / 'user_data' / 'server_settings.json'


# ----------------------------- Flow Progress API -----------------------------

@app.get('/flow/progress/{job_id}')
def flow_progress(job_id: str, steps: Optional[str] = None):
    """Best-effort progress estimation for flow jobs by scanning logs.
    Returns per-step status: pending/running/ok/failed.
    """
    # Fetch all logs for this job
    res = jobs.logs(job_id, 0)
    if isinstance(res, dict) and res.get('error'):
        return _error('JOB_NOT_FOUND', str(res.get('error')))
    running = bool(res.get('running'))
    code = res.get('code')
    returncode = res.get('returncode')
    raw_lines = [str(x) for x in (res.get('logs') or [])]
    lines = [x.lower() for x in raw_lines]
    steps_list = [s for s in (steps.split(',') if steps else ['feature','expression','ml','rl','backtest']) if s]

    # Prefer explicit flow markers printed by scripts/agent_flow.py
    # [FLOW] STEP_START <name> / STEP_OK <name> / STEP_FAIL <name>
    markers = { 'start': set(), 'ok': set(), 'fail': set() }
    phases: dict[str, str] = {}
    start_idx: dict[str, int] = {}
    # scan with index for dynamic estimates
    for idx, ln in enumerate(lines):
        if '[flow]' in ln and 'step_' in ln:
            # normalize
            if 'step_start' in ln:
                for nm in ['feature','expression','ml','rl','backtest']:
                    if f' {nm}' in ln:
                        markers['start'].add(nm)
                        if nm not in start_idx:
                            start_idx[nm] = idx
            elif 'step_ok' in ln:
                for nm in ['feature','expression','ml','rl','backtest']:
                    if f' {nm}' in ln:
                        markers['ok'].add(nm)
            elif 'step_fail' in ln:
                for nm in ['feature','expression','ml','rl','backtest']:
                    if f' {nm}' in ln:
                        markers['fail'].add(nm)
        if '[flow]' in ln and 'phase ' in ln:
            # [FLOW] PHASE <name> prepare|execute|summarize
            for nm in ['feature','expression','ml','rl','backtest']:
                if f' phase {nm} ' in ln:
                    if ' prepare' in ln:
                        phases[nm] = 'prepare'
                        if nm not in start_idx:
                            start_idx[nm] = idx
                    elif ' execute' in ln:
                        phases[nm] = 'execute'
                        if nm not in start_idx:
                            start_idx[nm] = idx
                    elif ' summarize' in ln:
                        phases[nm] = 'summarize'

    # Simple keyword map per step（回退）
    kw = {
        'feature': ['feature_agent', 'freqai_feature_agent', '--pairs', 'freqai features', 'feature file'],
        'expression': ['expression_agent', 'freqai_expression_agent', '--llm', 'llm', 'expressions'],
        'ml': ['training', 'model', 'lightgbm', 'xgboost', 'catboost', 'pytorch', 'training_summary.json'],
        'rl': ['rl', 'ppo', 'stable-baselines3', 'train_rl'],
        'backtest': ['backtesting', 'backtest', 'backtest-result', 'results', 'trades-'],
    }
    # Identify last seen step
    last_seen = -1
    total_lines = len(lines)
    for idx, name in enumerate(steps_list):
        keys = kw.get(name, [name])
        if any(any(k in ln for k in keys) for ln in lines):
            last_seen = max(last_seen, idx)
    items = []
    items = []
    for idx, name in enumerate(steps_list):
        # explicit markers win
        if name in markers['fail']:
            status = 'failed'
            phase = phases.get(name) or 'execute'
        elif name in markers['ok']:
            status = 'ok'
            phase = 'summarize'
        elif name in markers['start']:
            status = 'running'
            phase = phases.get(name) or 'execute'
        else:
            # fallback heuristic
            if last_seen < 0:
                status = 'running' if running and idx == 0 else 'pending'
                phase = 'prepare' if status == 'running' else None
            else:
                if idx < last_seen:
                    status = 'ok'
                    phase = 'summarize'
                elif idx == last_seen:
                    if running:
                        status = 'running'
                        phase = 'execute'
                    else:
                        status = 'ok' if (returncode == 0 and code == 'OK') else 'failed'
                        phase = 'summarize' if status == 'ok' else 'execute'
                else:
                    status = 'pending'
                    phase = None
        # percent mapping per-phase (with dynamic boost during execute based on log growth)
        percent = 0
        if status == 'pending':
            percent = 0
        elif status == 'running':
            if phase == 'prepare':
                percent = 10
            elif phase == 'execute':
                base = 50
                # dynamic increment based on lines from start
                sidx = start_idx.get(name, total_lines)
                exec_span = max(1, total_lines - sidx)
                # heuristic expected span
                expected = 200
                ratio = min(1.0, exec_span / expected)
                percent = int(base + 40 * ratio)
            elif phase == 'summarize':
                percent = 95
            else:
                percent = 10
        elif status == 'ok':
            percent = 100
        elif status == 'failed':
            percent = 50
        items.append({'name': name, 'status': status, 'phase': phase, 'percent': percent})
    return {'job_id': job_id, 'running': running, 'code': code, 'steps': items}


@app.get('/flow/stream/{job_id}')
def flow_stream(job_id: str, steps: Optional[str] = None):
    steps_list = [s for s in (steps.split(',') if steps else ['feature','expression','ml','rl','backtest']) if s]

    def _gen():
        last_next = 0
        # initial
        init_logs = jobs.logs(job_id, last_next)
        if isinstance(init_logs, dict) and init_logs.get('error'):
            err = _error('JOB_NOT_FOUND', str(init_logs.get('error')))
            yield f"event: progress\ndata: {_jsonmod.dumps(err, ensure_ascii=False)}\n\n"
            return
        last_next = int(init_logs.get('next') or last_next)
        payload = flow_progress(job_id, steps=(','.join(steps_list)))
        payload['delta'] = init_logs.get('logs') or []
        payload['next'] = last_next
        yield f"event: progress\ndata: {_jsonmod.dumps(payload, ensure_ascii=False)}\n\n"
        while True:
            data = jobs.logs(job_id, last_next)
            if isinstance(data, dict) and data.get('error'):
                err = _error('JOB_NOT_FOUND', str(data.get('error')))
                yield f"event: progress\ndata: {_jsonmod.dumps(err, ensure_ascii=False)}\n\n"
                break
            last_next = int(data.get('next') or last_next)
            payload = flow_progress(job_id, steps=(','.join(steps_list)))
            payload['delta'] = data.get('logs') or []
            payload['next'] = last_next
            yield f"event: progress\ndata: {_jsonmod.dumps(payload, ensure_ascii=False)}\n\n"
            if not data.get('running'):
                break
            time.sleep(1.0)

    return StreamingResponse(_gen(), media_type='text/event-stream')


@app.websocket('/flow/ws/{job_id}')
async def flow_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        last_next = 0
        while True:
            data = jobs.logs(job_id, last_next)
            if isinstance(data, dict) and data.get('error'):
                await websocket.send_json(_error('JOB_NOT_FOUND', str(data.get('error'))))
                break
            last_next = int(data.get('next') or last_next)
            payload = flow_progress(job_id)
            await websocket.send_json(payload)
            if not data.get('running'):
                break
            await asyncio.sleep(1.0)
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


def _load_settings() -> dict:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
    except Exception:
        pass
    # defaults from env
    return {
        'llm_base_url': os.environ.get('LLM_BASE_URL'),
        'llm_model': os.environ.get('LLM_MODEL'),
        'default_timeframe': '4h',
        'note': 'Server settings for Agent Market',
    }


def _save_settings(obj: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')



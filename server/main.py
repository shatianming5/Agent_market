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
import sys as _sys
if str(SRC) not in _sys.path:
    _sys.path.append(str(SRC))
import pandas as _pd  # noqa: E402
import json as _json  # noqa: E402
try:
    from agent_market.freqai.features import apply_configured_features as _apply_features  # type: ignore
except Exception:  # pragma: no cover - fallback if features module missing
    def _apply_features(df, cfg):
        return df

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
    # æ€§èƒ½ä¼˜åŒ–å¼€å…³ä¸Žå¯é€‰é˜ˆå€¼ï¼ˆä¸ä¼ åˆ™ä½¿ç”¨å¿«é€Ÿé»˜è®¤ï¼‰
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


class DownloadDataReq(BaseModel):
    config: str = Field(..., description="Path to freqtrade JSON config")
    timeframe: Optional[str] = Field(None, description="Single timeframe, e.g. '1h'")
    timeframes: Optional[list[str]] = Field(None, description="Multiple timeframes, e.g. ['1h','4h']")
    pairs: Optional[list[str]] = Field(None, description="Pairs list, e.g. ['BTC/USDT','ETH/USDT']")
    pairs_file: Optional[str] = Field(None, description="Path to pairs file")
    timerange: Optional[str] = Field(None, description="Timerange like 20200701-20210131")
    days: Optional[int] = Field(None, description="Limit to last N days")
    exchange: Optional[str] = Field(None, description="Exchange name to use")
    erase: bool = Field(False, description="Erase existing data before downloading")
    new_pairs: bool = Field(False, description="Only download pairs not present yet")
    dl_trades: bool = Field(False, description="Download trades data instead of OHLCV")
    prepend: bool = Field(False, description="Prepend earlier history if available")


class TrainMLReq(BaseModel):
    config_path: Optional[str] = Field(None, description="Path to training config JSON")
    config: Optional[dict] = Field(None, description="Inline training config object")


# --------------------------- Expressions Management ---------------------------


class ExpressionsPayload(BaseModel):
    expressions: list = Field(default_factory=list)


def _expressions_candidates() -> list[Path]:
    return [
        ROOT / 'user_data' / 'freqai_expressions.json',
        ROOT / 'freqtrade' / 'user_data' / 'freqai_expressions.json',
    ]


def _resolve_expressions_path() -> Path:
    for p in _expressions_candidates():
        if p.exists():
            return p
    # default to ROOT/user_data
    target = ROOT / 'user_data' / 'freqai_expressions.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@app.get('/expressions')
def expressions_get():
    path = _resolve_expressions_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
        else:
            data = {'expressions': []}
    except json.JSONDecodeError:
        return {"status": "error", "code": "MALFORMED", "message": f"Expressions JSON malformed: {path}"}
    return {"path": str(path.relative_to(ROOT)), "expressions": data.get('expressions', [])}


@app.get('/expressions/allowed')
def expressions_allowed():
    allowed_funcs = ['z','abs','sign','clip','shift','roll_mean','roll_std','pct_change','ema','rolling_max','rolling_min','log1p','tanh']
    base_cols = ['date','open','high','low','close','volume','atr_pct_14','ema_fast_55','ema_slow_200','trend_score','trend_up','prediction','prediction_z']
    return {"functions": allowed_funcs, "base_columns": base_cols}


@app.put('/expressions')
def expressions_put(payload: ExpressionsPayload = Body(...)):
    path = _resolve_expressions_path()
    try:
        obj = {'expressions': list(payload.expressions or [])}
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as exc:
        return {"status": "error", "code": "WRITE_FAILED", "message": str(exc)}
    return {"status": "ok", "path": str(path.relative_to(ROOT)), "count": len(payload.expressions or [])}


@app.post('/expressions')
def expressions_post(item: dict = Body(...)):
    path = _resolve_expressions_path()
    try:
        data = {'expressions': []}
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
        items = list(data.get('expressions', []))
        items.append(item)
        data['expressions'] = items
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        return {"status": "ok", "index": len(items) - 1}
    except Exception as exc:
        return {"status": "error", "code": "WRITE_FAILED", "message": str(exc)}


@app.patch('/expressions/{index}')
def expressions_patch(index: int, item: dict = Body(...)):
    path = _resolve_expressions_path()
    try:
        data = {'expressions': []}
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
        items = list(data.get('expressions', []))
        if index < 0 or index >= len(items):
            return {"status": "error", "code": "INDEX_OUT_OF_RANGE"}
        items[index] = item
        data['expressions'] = items
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "code": "WRITE_FAILED", "message": str(exc)}


@app.delete('/expressions/{index}')
def expressions_delete(index: int):
    path = _resolve_expressions_path()
    try:
        data = {'expressions': []}
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
        items = list(data.get('expressions', []))
        if index < 0 or index >= len(items):
            return {"status": "error", "code": "INDEX_OUT_OF_RANGE"}
        removed = items.pop(index)
        data['expressions'] = items
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        return {"status": "ok", "removed": removed}
    except Exception as exc:
        return {"status": "error", "code": "WRITE_FAILED", "message": str(exc)}


def _load_feature_cfg() -> dict:
    for cand in [ROOT / 'user_data' / 'freqai_features.json', ROOT / 'freqtrade' / 'user_data' / 'freqai_features.json']:
        if cand.exists():
            try:
                return _json.loads(cand.read_text(encoding='utf-8'))
            except _json.JSONDecodeError:
                continue
    return {'features': []}


def _load_pair_df(config: Optional[str], pair: str, timeframe: str):
    cfg = _load_freqtrade_config(config)
    datadir = Path(cfg.get('datadir') or 'freqtrade/user_data/data')
    exch = (cfg.get('exchange', {}) or {}).get('name') or 'binanceus'
    base = (ROOT / datadir).resolve() / exch
    path = base / f"{pair.replace('/','_')}-{timeframe}.feather"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = _pd.read_feather(path)
    df['date'] = _pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


@app.post('/expressions/preview')
def expressions_preview(
    pair: str = Body(...),
    timeframe: str = Body(...),
    expression: str = Body(...),
    config: Optional[str] = Body(None),
    apply_features: bool = Body(True),
):
    try:
        df = _load_pair_df(config, pair, timeframe)
    except FileNotFoundError as exc:
        return {"status": "error", "code": "NO_DATA", "message": str(exc)}
    feats = _load_feature_cfg() if apply_features else {'features': []}
    try:
        df2 = _apply_features(df.copy(), feats)
    except Exception:
        df2 = df.copy()
    # safe eval
    allowed = {
        'z': lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9),
        'abs': __import__('numpy').abs,
        'sign': __import__('numpy').sign,
        'clip': __import__('numpy').clip,
        'shift': lambda s, n=1: s.shift(int(n)),
        'roll_mean': lambda s, w=3: s.rolling(int(w)).mean(),
        'roll_std': lambda s, w=3: s.rolling(int(w)).std(ddof=0),
        'pct_change': lambda s, n=1: s.pct_change(int(n)),
        'ema': lambda s, span=5: s.ewm(span=int(span), adjust=False).mean(),
        'rolling_max': lambda s, w=5: s.rolling(int(w)).max(),
        'rolling_min': lambda s, w=5: s.rolling(int(w)).min(),
        'log1p': lambda s: __import__('numpy').log1p(s),
        'tanh': lambda s: __import__('numpy').tanh(s),
    }
    local_dict = {col: df2[col] for col in df2.columns if col not in ('date',)}
    local_dict.update(allowed)
    try:
        res = eval(expression, {'__builtins__': {}}, local_dict)
    except Exception as exc:
        return {"status": "error", "code": "EVAL_FAILED", "message": str(exc)}
    import numpy as _np
    s = _pd.Series(res, index=df2.index).astype(float).replace([_np.inf, -_np.inf], _np.nan)
    s = s.ffill().bfill()
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    q = {p: float(s.quantile(p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    zq = {p: float(z.quantile(p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    resp = {
        'count': int(s.count()),
        'mean': float(s.mean()),
        'std': float(s.std(ddof=0)),
        'min': float(s.min()) if s.size else None,
        'max': float(s.max()) if s.size else None,
        'quantiles': q,
        'z_quantiles': zq,
        'head': [float(x) if _np.isfinite(x) else None for x in s.head(5).values.tolist()],
        'tail': [float(x) if _np.isfinite(x) else None for x in s.tail(5).values.tolist()],
    }
    return resp


# --------------------------- Data Summary APIs ---------------------------


def _load_freqtrade_config(cfg: Optional[str]) -> dict:
    if not cfg:
        return {}
    p = Path(cfg)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


@app.get('/data/summary')
def data_summary(
    config: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframes: Optional[str] = None,
):
    cfg = _load_freqtrade_config(config)
    datadir = Path(cfg.get('datadir') or 'freqtrade/user_data/data')
    exch = exchange or (cfg.get('exchange', {}) or {}).get('name') or 'binanceus'
    datadir = (ROOT / datadir).resolve() if not datadir.is_absolute() else datadir
    base = datadir / exch
    if not base.exists():
        return {"exchange": exch, "timeframes": {}, "message": f"Data directory missing: {base}"}
    tfs = []
    if timeframes:
        tfs = [t.strip() for t in timeframes.split(',') if t.strip()]
    else:
        # Infer timeframes from files
        tfs = sorted({f.suffix.replace('.feather','').split('-')[-1] for f in base.glob('*.feather')})
    import pandas as pd
    res: dict[str, list[dict]] = {}
    for tf in tfs:
        rows = []
        for f in base.glob(f'*-{tf}.feather'):
            try:
                df = pd.read_feather(f, columns=['date'])
                rows.append({
                    'pair': f.stem.replace(f'-{tf}', '').replace('_','/'),
                    'count': int(df.shape[0]),
                    'start': df['date'].min().isoformat() if not df.empty else None,
                    'end': df['date'].max().isoformat() if not df.empty else None,
                })
            except Exception:
                rows.append({'pair': f.stem.replace(f'-{tf}', '').replace('_','/'), 'count': 0, 'start': None, 'end': None})
        res[tf] = rows
    return {"exchange": exch, "root": str(base), "timeframes": res}


@app.get('/data/check-missing')
def data_check_missing(
    config: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframes: Optional[str] = None,
    pairs: Optional[str] = None,
    timerange: Optional[str] = None,
):
    from datetime import datetime
    cfg = _load_freqtrade_config(config)
    datadir = Path(cfg.get('datadir') or 'freqtrade/user_data/data')
    exch = exchange or (cfg.get('exchange', {}) or {}).get('name') or 'binanceus'
    datadir = (ROOT / datadir).resolve() if not datadir.is_absolute() else datadir
    base = datadir / exch
    want_pairs = [p.strip() for p in (pairs or '').split(',') if p.strip()]
    want_tfs = [t.strip() for t in (timeframes or '').split(',') if t.strip()]
    def parse_tr(tr: Optional[str]):
        if not tr:
            return None, None
        try:
            s, e = tr.split('-')
            sdt = datetime.strptime(s, '%Y%m%d')
            edt = datetime.strptime(e, '%Y%m%d')
            return sdt, edt
        except Exception:
            return None, None
    start, end = parse_tr(timerange)
    import pandas as pd
    missing = []
    insufficient = []
    for tf in want_tfs:
        for pair in want_pairs:
            f = base / f"{pair.replace('/','_')}-{tf}.feather"
            if not f.exists():
                missing.append({'pair': pair, 'timeframe': tf, 'reason': 'no_file'})
                continue
            try:
                df = pd.read_feather(f, columns=['date'])
            except Exception:
                missing.append({'pair': pair, 'timeframe': tf, 'reason': 'read_error'})
                continue
            if df.empty:
                missing.append({'pair': pair, 'timeframe': tf, 'reason': 'empty'})
                continue
            if start is not None and end is not None:
                smin = pd.to_datetime(df['date'].min()).to_pydatetime()
                smax = pd.to_datetime(df['date'].max()).to_pydatetime()
                if smin > start or smax < end:
                    insufficient.append({'pair': pair, 'timeframe': tf, 'file_start': smin.isoformat(), 'file_end': smax.isoformat(), 'want_start': start.isoformat(), 'want_end': end.isoformat()})
    return {"missing": missing, "insufficient": insufficient}


# --------------------------- Strategy Params & Backtest Summary ---------------------------


def _strategy_params_path() -> Path:
    p = ROOT / 'freqtrade' / 'user_data' / 'strategies' / 'ExpressionLongStrategy.json'
    if p.exists():
        return p
    return ROOT / 'freqtrade' / 'user_data' / 'strategies' / 'ExpressionLongStrategy.json'


@app.get('/strategy/params')
def strategy_params_get():
    path = _strategy_params_path()
    try:
        data = _json.loads(path.read_text(encoding='utf-8'))
        return {'path': str(path.relative_to(ROOT)), 'data': data}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


@app.put('/strategy/params')
def strategy_params_put(payload: dict = Body(...)):
    path = _strategy_params_path()
    try:
        path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'status': 'ok', 'path': str(path.relative_to(ROOT))}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


def _find_latest_backtest_zip(results_dir: Path) -> Optional[Path]:
    results_dir = results_dir.resolve()
    if not results_dir.exists():
        return None
    last_path = results_dir / '.last_result.json'
    if last_path.exists():
        try:
            latest_name = _json.loads(last_path.read_text(encoding='utf-8')).get('latest_backtest')
            if latest_name:
                candidate = results_dir / latest_name
                if candidate.exists():
                    return candidate
        except _json.JSONDecodeError:
            pass
    zips = list(results_dir.glob('backtest-result-*.zip'))
    if not zips:
        return None
    return max(zips, key=lambda p: p.stat().st_mtime)


@app.get('/backtest/summary/latest')
def backtest_summary_latest():
    results_dir = ROOT / 'user_data' / 'backtest_results'
    z = _find_latest_backtest_zip(results_dir)
    if z is None:
        return {'status': 'error', 'message': 'No results'}
    import zipfile
    try:
        with zipfile.ZipFile(z) as zf:
            name = next((n for n in zf.namelist() if n.endswith('.json') and '_config' not in n), None)
            if not name:
                return {'status': 'error', 'message': 'No result JSON in archive'}
            data = _json.loads(zf.read(name))
        strategy_block = data.get('strategy', {})
        if not strategy_block:
            return {'status': 'error', 'message': 'Missing strategy block'}
        strat_name, metrics = next(iter(strategy_block.items()))
        comparison = data.get('strategy_comparison', [])
        return {
            'source': z.name,
            'strategy': strat_name,
            'metrics': metrics,
            'comparison': comparison,
        }
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


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
    # é€Ÿåº¦ä¼˜å…ˆï¼šé™åˆ¶å€™é€‰è§„æ¨¡ï¼Œå…³é—­ç»„åˆç‰¹å¾ï¼Œé™ä½Žç¨³å®šæ€§çª—å£
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
    # ä»…ç¡®ä¿ src/ å¯å¯¼å…¥ï¼ˆé¿å…è¦†ç›– conda å†…å·²å®‰è£…çš„ freqtrade åŒ…ï¼‰
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
    # çŸ­æ—¶é—´èŒƒå›´ä¼˜å…ˆï¼ˆé™¤éžæ˜¾å¼æŒ‡å®š fast=Falseï¼‰
    fast_mode = req.fast or (req.force_fast if req.force_fast is not None else SETTINGS.force_fast)
    if fast_mode:
        timerange_value = req.timerange or '20200701-20210131'
    else:
        timerange_value = req.timerange
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


@app.post("/run/download-data")
def run_download_data(req: DownloadDataReq = Body(...)):
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}

    env_py = _conda_env_python('freqtrade')
    conda_prefix = _conda_prefix_args() if not env_py else None
    py = env_py or ('python' if conda_prefix else sys.executable)

    from shutil import which
    if env_py:
        base = [py, '-m', 'freqtrade', 'download-data']
    else:
        use_bin = which('freqtrade') is not None and conda_prefix is None
        if use_bin:
            base = ['freqtrade', 'download-data']
        else:
            base = [py, '-m', 'freqtrade', 'download-data']

    cmd = (conda_prefix or []) + base + ['--config', str(cfg_path)]

    # Timeframes
    tfs = []
    if req.timeframes:
        tfs = [str(t) for t in req.timeframes if t]
    elif req.timeframe:
        tfs = [str(req.timeframe)]
    for tf in tfs:
        cmd += ['-t', tf]

    # Pairs
    if req.pairs_file:
        p = Path(req.pairs_file)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        cmd += ['--pairs-file', str(p)]
    if req.pairs:
        for pair in req.pairs:
            cmd += ['-p', str(pair)]

    # Other options
    if req.exchange:
        cmd += ['--exchange', req.exchange]
    if req.timerange:
        cmd += ['--timerange', req.timerange]
    if isinstance(req.days, int) and req.days > 0:
        cmd += ['--days', str(int(req.days))]
    if req.erase:
        cmd += ['--erase']
    if req.new_pairs:
        cmd += ['--new-pairs']
    if req.dl_trades:
        cmd += ['--dl-trades']
    if req.prepend:
        cmd += ['--prepend']

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(SRC), str(ROOT / 'freqtrade'), env.get('PYTHONPATH', '')])
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.post("/run/train-ml")
def run_train_ml(req: TrainMLReq = Body(...)):
    # Accept either config_path or inline config
    cfg_path: Optional[Path] = None
    if req.config_path:
        cfg_path = Path(req.config_path)
        if not cfg_path.is_absolute():
            cfg_path = (ROOT / cfg_path).resolve()
        if not cfg_path.exists():
            return {"status": "error", "code": "TRAIN_CONFIG_NOT_FOUND", "message": f"Training config not found: {cfg_path}"}
    elif req.config is not None:
        # Write to a tmp file
        tmp_dir = ROOT / 'user_data' / 'tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        import uuid, json
        cfg_path = tmp_dir / f"train_{uuid.uuid4().hex[:8]}.json"
        cfg_path.write_text(json.dumps(req.config, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        return {"status": "error", "code": "TRAIN_CONFIG_MISSING", "message": "Provide config_path or config"}

    env_py = _conda_env_python('freqtrade')
    conda_prefix = _conda_prefix_args() if not env_py else None
    py = env_py or ('python' if conda_prefix else sys.executable)

    script = ROOT / 'scripts' / 'train_ml.py'
    if not script.exists():
        return {"status": "error", "code": "TRAIN_SCRIPT_MISSING", "message": f"Script not found: {script}"}
    cmd = (conda_prefix or []) + [py, str(script), '--config', str(cfg_path)]
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
@app.get('/jobs/{job_id}/progress')
def job_progress(job_id: str):
    status = jobs.status(job_id)
    if 'error' in status:
        return status
    data = jobs.logs(job_id, 0)
    logs = data.get('logs', [])
    current=0; total=0; label=''
    import re, datetime as _dt
    pat = re.compile(r"^\[STEP\]\s+\d{2}:\d{2}:\d{2}\s+(?:\[(\d+)/(\d+)\]\s+([^\n]+)|(.+))$")
    for line in logs:
        m = pat.match(line)
        if not m:
            continue
        if m.group(1):
            current = int(m.group(1) or 0)
            total = int(m.group(2) or 0)
            label = (m.group(3) or '').strip()
    elapsed=None
    try:
        if status.get('started_at'):
            t0 = _dt.datetime.fromisoformat(status['started_at'].replace('Z','+00:00'))
            elapsed = int((_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds())
    except Exception:
        pass
    pct = int(min(100, (current/total*100))) if total else 0
    return {
        'id': status.get('id'),
        'running': status.get('running'),
        'started_at': status.get('started_at'),
        'finished_at': status.get('finished_at'),
        'current': current,
        'total': total,
        'label': label,
        'percent': pct,
        'elapsed': elapsed,
    }


@app.post('/jobs/dev/sleep')
def job_dev_sleep(secs: int = 15):
    if not os.environ.get('APP_DEV_JOBS'):
        return {"error": "dev jobs disabled"}
    py = sys.executable
    code = f"import time; [print(f'line{{i}}') or time.sleep(1) for i in range({int(secs)})]"
    cmd = [py, '-c', code]
    env = os.environ.copy()
    job_id = jobs.start(cmd, cwd=ROOT, env=env)
    return {"status": "started", "job_id": job_id}



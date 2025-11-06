from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
from fastapi import FastAPI, Body, Depends, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse, FileResponse

from .job_manager import JobManager
from .config import load_settings, Settings
from .db import DB
from . import routes_agents, routes_orders, routes_triggers, routes_secrets, routes_connectors
from .triggers import TriggerManager
from .secrets import SecretsManager
from .connector_service import ConnectorService
from .telemetry import configure_logging, configure_tracing

from agent_market.paths import RUNS_DIR, SRC_ROOT, USER_DATA_DIR, PROJECT_ROOT

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("AGENT_MARKET_RUNS_DIR", str(RUNS_DIR))).resolve()
EXPRESSIONS_FILE = USER_DATA_DIR / "freqai_expressions.json"
FEATURES_FILE = USER_DATA_DIR / "freqai_features_multi.json"
BACKTEST_RESULTS_DIR = USER_DATA_DIR / "backtest_results"
TMP_DIR = USER_DATA_DIR / "tmp"
APP_ROOT = PROJECT_ROOT / "app"
SCRIPTS_DIR = APP_ROOT / "scripts"
FREQTRADE_DIR = APP_ROOT / "freqtrade"
FREQTRADE_PKG_DIR = FREQTRADE_DIR / "freqtrade"
import sys as _sys
if str(SRC_ROOT) not in _sys.path:
    _sys.path.append(str(SRC_ROOT))
import pandas as _pd  # noqa: E402
import json as _json  # noqa: E402
import json  # ensure legacy usages still resolve
try:
    from agent_market.freqai.features import apply_configured_features as _apply_features  # type: ignore
except Exception:  # pragma: no cover - fallback if features module missing
    def _apply_features(df, cfg):
        return df


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    cleaned: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if cleaned:
                cleaned.pop()
            continue
        cleaned.append(part)
    candidate = PROJECT_ROOT.joinpath(*cleaned)
    return candidate.resolve()


def _augment_pythonpath(env: dict, *extra: Path) -> None:
    entries = [str(SRC_ROOT)]
    if FREQTRADE_PKG_DIR.exists():
        entries.append(str(FREQTRADE_PKG_DIR.resolve()))
    for item in extra:
        entries.append(str(item))
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)


SETTINGS = load_settings()

configure_logging(SETTINGS)

app = FastAPI(title=SETTINGS.api_title, version=SETTINGS.api_version)
app.add_middleware(CORSMiddleware, allow_origins=SETTINGS.allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
configure_tracing(app, SETTINGS)
def _on_job_step(job_id: str, idx: int, total: int, label: str, ts: str) -> None:
    try:
        DB_HANDLE.log_job_step(job_id, idx, total, label, ts)
    except Exception:
        pass

jobs = JobManager(max_seconds=SETTINGS.job_max_seconds, on_step=_on_job_step)

# Initialize database
DB_HANDLE = DB(SETTINGS.db_path)
DB_HANDLE.init()
TRIGGER_MANAGER = TriggerManager(DB_HANDLE, jobs)
SECRETS_MANAGER = SecretsManager(DB_HANDLE, SETTINGS)
CONNECTOR_SERVICE = ConnectorService(DB_HANDLE, SECRETS_MANAGER, jobs=jobs, triggers=TRIGGER_MANAGER)


def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _load_run_manifest(run_id: str) -> tuple[Dict[str, Any], Path]:
    run_dir = _run_dir(run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found for run {run_id}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.setdefault("run_id", run_id)
    data.setdefault("run_dir", _safe_relative(run_dir))
    return data, run_dir


def _list_run_manifests() -> List[Dict[str, Any]]:
    if not RUNS_ROOT.exists():
        return []
    items: List[Dict[str, Any]] = []
    for manifest_path in sorted(RUNS_ROOT.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        run_id = manifest_path.parent.name
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "run_id": run_id,
                "status": data.get("status"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "run_dir": str(manifest_path.parent),
                "manifest_path": str(manifest_path.relative_to(manifest_path.parent)),
            }
        )
    return items

# Dependency override providers for routers
def _get_db() -> DB:
    return DB_HANDLE

def _get_trigger_manager() -> TriggerManager:
    return TRIGGER_MANAGER

def _get_secrets_manager() -> SecretsManager:
    return SECRETS_MANAGER

def _get_connector_service() -> ConnectorService:
    return CONNECTOR_SERVICE

app.dependency_overrides[routes_agents.get_db] = _get_db
app.dependency_overrides[routes_orders.get_db] = _get_db
app.dependency_overrides[routes_triggers.get_db] = _get_db
app.dependency_overrides[routes_triggers.get_manager] = _get_trigger_manager
app.dependency_overrides[routes_secrets.get_manager] = _get_secrets_manager
app.dependency_overrides[routes_secrets.get_settings] = lambda: SETTINGS
app.dependency_overrides[routes_connectors.get_service] = _get_connector_service

app.include_router(routes_agents.router)
app.include_router(routes_orders.router)
app.include_router(routes_triggers.router)
app.include_router(routes_secrets.router)
app.include_router(routes_connectors.router)


@app.on_event("startup")
def _startup_triggers() -> None:
    try:
        TRIGGER_MANAGER.start()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to start trigger manager: %s", exc)


@app.on_event("shutdown")
def _shutdown_triggers() -> None:
    try:
        TRIGGER_MANAGER.stop()
    except Exception:
        pass


def _conda_prefix_args() -> list[str] | None:
    """Return ['conda','run','--no-capture-output','-n','freqtrade'] if conda env 'freqtrade' exists.

    Falls back to None if conda not available or env missing.
    """
    disable = os.environ.get("AGENT_DISABLE_CONDA", "").strip().lower()
    if disable in {"1", "true", "yes", "on"}:
        return None
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
    disable = os.environ.get("AGENT_DISABLE_CONDA", "").strip().lower()
    if disable in {"1", "true", "yes", "on"}:
        return None
    try:
        import subprocess
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
    output: str = Field("resources/user_data/freqai_expressions.json")
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
    strategy_path: str = Field("app/freqtrade/user_data/strategies")
    timerange: str = Field("20210101-20211231")
    freqaimodel: str = Field("LightGBMRegressor")
    export: bool = True
    export_filename: str = Field("resources/user_data/backtest_results/latest_trades_multi")
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


def _resolve_expressions_path() -> Path:
    EXPRESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    return EXPRESSIONS_FILE


def _safe_relative(path: Path) -> str:
    for base in (ROOT, ROOT.parent, PROJECT_ROOT):
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            continue
    return str(path.resolve())


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
    return {"path": _safe_relative(path), "expressions": data.get('expressions', [])}


@app.get('/expressions/allowed')
def expressions_allowed():
    allowed_funcs = ['z','abs','sign','clip','shift','roll_mean','roll_std','pct_change','ema','rolling_max','rolling_min','rolling','log1p','tanh']
    base_cols = ['date','open','high','low','close','volume','atr_pct_14','ema_fast_55','ema_slow_200','trend_score','trend_up','prediction','prediction_z']
    return {"functions": allowed_funcs, "base_columns": base_cols}


@app.post('/expressions/validate')
def expressions_validate(expression: str = Body(...)):
    # Very light static validation: flag illegal chars and unknown functions
    import re
    allowed_funcs = {'z','abs','sign','clip','shift','roll_mean','roll_std','pct_change','ema','rolling_max','rolling_min','log1p','tanh'}
    illegal = re.findall(r"[^A-Za-z0-9_\s\(\)\+\-\*/%,\.:<>!=]", expression or '')
    # Find identifiers (function candidates)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression or '')
    # Common math names to ignore
    ignore = {'e','pi'}
    funcs = sorted({t for t in tokens if t.lower() in allowed_funcs})
    unknown_funcs = sorted({t for t in tokens if t.lower() not in allowed_funcs and t not in ignore})
    return {
        'illegal_chars': list(dict.fromkeys(illegal)),
        'allowed_funcs': sorted(list(allowed_funcs)),
        'used_funcs': funcs,
        'unknown_identifiers': unknown_funcs,
    }


@app.put('/expressions')
def expressions_put(payload: ExpressionsPayload = Body(...)):
    path = _resolve_expressions_path()
    try:
        obj = {'expressions': list(payload.expressions or [])}
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        return {"status": "ok", "path": _safe_relative(path), "count": len(payload.expressions or [])}
    except Exception as exc:
        return {"status": "error", "code": "WRITE_FAILED", "message": str(exc)}


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
    if FEATURES_FILE.exists():
        try:
            return _json.loads(FEATURES_FILE.read_text(encoding='utf-8'))
        except _json.JSONDecodeError:
            return {'features': []}
    return {'features': []}


def _load_pair_df(config: Optional[str], pair: str, timeframe: str):
    cfg = _load_freqtrade_config(config)
    datadir_val = cfg.get('datadir') or 'resources/user_data/data'
    datadir = Path(datadir_val)
    if not datadir.is_absolute():
        datadir = _resolve_project_path(datadir)
    exch = (cfg.get('exchange', {}) or {}).get('name') or 'binanceus'
    base = datadir / exch
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
        # alias: generic rolling -> mean for convenience in预览/提示
        'rolling': lambda s, w=5: s.rolling(int(w)).mean(),
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


@app.get('/features')
def features_get(config_path: Optional[str] = None):
    # Load active features JSON from default path
    path = FEATURES_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rel = _safe_relative(path)
        except ValueError:
            rel = path
        return {"path": str(rel), "features": []}
    try:
        data = _json.loads(path.read_text(encoding='utf-8'))
        try:
            rel = _safe_relative(path)
        except ValueError:
            rel = path
        return {"path": str(rel), "features": data.get('features', [])}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post('/config/use-features-file')
def use_features_file(path: str = Body(..., embed=True)):
    src = _resolve_project_path(path)
    if not src.exists():
        return {"status": "error", "message": f"File not found: {src}"}
    try:
        payload = _json.loads(src.read_text(encoding='utf-8'))
        # validate minimal structure
        if not isinstance(payload, dict):
            return {"status": "error", "message": "Invalid features JSON"}
        dest = FEATURES_FILE
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return {"status": "ok", "active": _safe_relative(dest)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post('/config/use-expressions-file')
def use_expressions_file(path: str = Body(..., embed=True)):
    src = _resolve_project_path(path)
    if not src.exists():
        return {"status": "error", "message": f"File not found: {src}"}
    try:
        payload = _json.loads(src.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or 'expressions' not in payload:
            return {"status": "error", "message": "Invalid expressions JSON"}
        dest = _resolve_expressions_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return {"status": "ok", "active": _safe_relative(dest)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# --------------------------- Data Summary APIs ---------------------------


def _load_freqtrade_config(cfg: Optional[str]) -> dict:
    if not cfg:
        return {}
    p = _resolve_project_path(cfg)
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
    datadir = Path(cfg.get('datadir') or 'resources/user_data/data')
    exch = exchange or (cfg.get('exchange', {}) or {}).get('name') or 'binanceus'
    if not datadir.is_absolute():
        datadir = _resolve_project_path(datadir)
    base = datadir / exch
    if not base.exists():
        return {"exchange": exch, "timeframes": {}, "message": f"Data directory missing: {base}"}
    tfs = []
    if timeframes:
        tfs = [t.strip() for t in timeframes.split(',') if t.strip()]
    else:
        # Infer timeframes from filenames like PAIR-TF.feather
        tfs = sorted({(f.stem.rsplit('-', 1)[-1]) for f in base.glob('*.feather') if '-' in f.stem})
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
    datadir = Path(cfg.get('datadir') or 'resources/user_data/data')
    exch = exchange or (cfg.get('exchange', {}) or {}).get('name') or 'binanceus'
    if not datadir.is_absolute():
        datadir = _resolve_project_path(datadir)
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
    try:
        missing = []
        insufficient = []
        for tf in want_tfs:
            for pair in want_pairs:
                f = base / f"{pair.replace('/','_')}-{tf}.feather"
                if not f.exists():
                    missing.append({'pair': pair, 'timeframe': tf, 'reason': 'no_file', 'suggestion': 'new_pairs'})
                    continue
                try:
                    df = pd.read_feather(f, columns=['date'])
                except Exception as exc:
                    missing.append({'pair': pair, 'timeframe': tf, 'reason': f'read_error:{exc}', 'suggestion': 'erase'})
                    continue
                if df.empty:
                    missing.append({'pair': pair, 'timeframe': tf, 'reason': 'empty', 'suggestion': 'erase'})
                    continue
                if start is not None and end is not None:
                    smin_ts = pd.to_datetime(df['date'].min())
                    smax_ts = pd.to_datetime(df['date'].max())
                    try:
                        smin_ts = smin_ts.tz_convert(None)  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            smin_ts = smin_ts.tz_localize(None)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    try:
                        smax_ts = smax_ts.tz_convert(None)  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            smax_ts = smax_ts.tz_localize(None)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    smin = smin_ts.to_pydatetime()
                    smax = smax_ts.to_pydatetime()
                    if smin > start or smax < end:
                        insufficient.append({
                            'pair': pair,
                            'timeframe': tf,
                            'file_start': smin.isoformat(),
                            'file_end': smax.isoformat(),
                            'want_start': start.isoformat(),
                            'want_end': end.isoformat(),
                            'missing_earlier': bool(smin > start),
                            'missing_later': bool(smax < end),
                            'suggestion': 'prepend' if smin > start else 'none'
                        })
        return {"missing": missing, "insufficient": insufficient}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "exchange": exch, "base": str(base), "pairs": want_pairs, "timeframes": want_tfs}


@app.get('/data/columns')
def data_columns(pair: str, timeframe: str, config: Optional[str] = None, apply_features: bool = True):
    try:
        df = _load_pair_df(config, pair, timeframe)
    except FileNotFoundError as exc:
        return {"status": "error", "code": "NO_DATA", "message": str(exc)}
    feats = _load_feature_cfg() if apply_features else {'features': []}
    try:
        df2 = _apply_features(df.copy(), feats)
    except Exception:
        df2 = df
    cols = [c for c in df2.columns if c != 'date']
    return {"columns": cols}


# --------------------------- Strategy Params & Backtest Summary ---------------------------


def _strategy_params_path() -> Path:
    path = USER_DATA_DIR / 'strategies' / 'ExpressionLongStrategy.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@app.get('/strategy/params')
def strategy_params_get():
    path = _strategy_params_path()
    try:
        if not path.exists():
            return {'path': _safe_relative(path), 'data': {}}
        data = _json.loads(path.read_text(encoding='utf-8'))
        return {'path': _safe_relative(path), 'data': data}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


@app.put('/strategy/params')
def strategy_params_put(payload: dict = Body(...)):
    path = _strategy_params_path()
    try:
        path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'status': 'ok', 'path': _safe_relative(path)}
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
    results_dir = BACKTEST_RESULTS_DIR
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
        # Try to build per-pair summary
        pairs = []
        try:
            pr = metrics.get('pair_results')
            if isinstance(pr, list) and pr:
                for r in pr:
                    pairs.append({'pair': r.get('pair') or r.get('key'), 'trades': r.get('trades'), 'profit_total_pct': r.get('profit_total_pct')})
            else:
                # Fallback: derive from trades list if present
                tlist = metrics.get('trades')
                if isinstance(tlist, list):
                    agg = {}
                    for t in tlist:
                        key = t.get('pair')
                        if not key:
                            continue
                        a = agg.setdefault(key, {'pair': key, 'trades': 0, 'profit_total_pct': 0.0})
                        a['trades'] += 1
                        pratio = t.get('profit_ratio')
                        if isinstance(pratio, (int, float)):
                            a['profit_total_pct'] += float(pratio) * 100.0
                    pairs = sorted(agg.values(), key=lambda x: x['pair'])
        except Exception:
            pairs = []
        comparison = data.get('strategy_comparison', [])
        return {
            'source': z.name,
            'strategy': strat_name,
            'metrics': metrics,
            'pairs': pairs,
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


@app.post("/triggers/webhook/{trigger_id}")
async def trigger_webhook(trigger_id: str, request: Request) -> Dict[str, str]:
    try:
        payload = await request.json()
    except Exception:
        body_bytes = await request.body()
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            payload = {"raw": body_bytes.decode("utf-8", errors="ignore")}
    headers = {k.lower(): v for k, v in request.headers.items()}
    try:
        job_id = TRIGGER_MANAGER.fire_webhook(trigger_id, payload, headers)
    except KeyError:
        raise HTTPException(status_code=404, detail="trigger not found") from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id}


@app.post("/run/expression")
def run_expression(req: ExpressionReq = Body(...)):
    # Resolve and validate important paths
    cfg_path = _resolve_project_path(req.config)
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    ff_path = _resolve_project_path(req.feature_file)
    if not ff_path.exists():
        return {"status": "error", "code": "FEATURE_FILE_NOT_FOUND", "message": f"Feature file not found: {ff_path}"}
    conda_prefix = _conda_prefix_args()
    env_py = _conda_env_python('freqtrade')
    py = env_py or ('python' if conda_prefix else sys.executable)
    app_root = APP_ROOT
    wrapper_candidates = [
        SCRIPTS_DIR / 'expr_agent_wrapper.py',
        app_root / 'scripts' / 'expr_agent_wrapper.py',
    ]
    wrapper = next((p for p in wrapper_candidates if p.exists()), None)
    script_candidates = [
        FREQTRADE_PKG_DIR / 'scripts' / 'freqai_expression_agent.py',
        app_root / 'freqtrade' / 'freqtrade' / 'scripts' / 'freqai_expression_agent.py',
    ]
    script_path = next((p for p in script_candidates if p.exists()), None)
    if wrapper is not None:
        base = [py, str(wrapper)]
        feature_file_arg = str(ff_path)
    elif script_path is not None:
        base = [py, str(script_path)]
        feature_file_arg = str(ff_path)
    else:
        # fallback to module invocation
        base = [py, '-m', 'freqtrade.scripts.freqai_expression_agent']
        feature_file_arg = str(ff_path)
    output_path = _resolve_project_path(req.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info("Expression output resolved to %s", output_path)

    feedback_path: Optional[Path] = None
    if req.feedback:
        feedback_path = _resolve_project_path(req.feedback)

    cmd = (conda_prefix or []) + base
    cmd += [
        '--config', str(cfg_path),
        '--feature-file', feature_file_arg,
        '--output', str(output_path),
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
    if feedback_path:
        cmd += ['--feedback', str(feedback_path)]

    env = os.environ.copy()
    _augment_pythonpath(env)
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
    job_id = jobs.start(cmd, cwd=APP_ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.post("/run/backtest")
def run_backtest(req: BacktestReq = Body(...)):
    # Resolve and validate config and strategy path
    cfg_path = _resolve_project_path(req.config)
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    spath = Path(req.strategy_path) if req.strategy_path else None
    if spath and not spath.is_absolute():
        spath = _resolve_project_path(spath)
    if spath and not spath.exists():
        return {"status": "error", "code": "STRATEGY_PATH_NOT_FOUND", "message": f"Strategy path not found: {spath}"}

    env_py = _conda_env_python('freqtrade')
    conda_prefix = _conda_prefix_args() if not env_py else None
    py = env_py or ('python' if conda_prefix else sys.executable)
    binary = 'freqtrade'
    from shutil import which
    # Prefer direct env python if available to avoid "conda run" stdio issues
    wrapper_bt = SCRIPTS_DIR / 'backtest_wrapper.py'
    if wrapper_bt.exists():
        base = [py, str(wrapper_bt)]
    elif env_py:
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
    export_path: Optional[Path] = None
    if req.export:
        export_path = _resolve_project_path(req.export_filename)
        export_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = (conda_prefix or []) + base + [
        '--config', str(cfg_path),
        '--strategy', req.strategy,
        '--strategy-path', str(spath) if spath else req.strategy_path,
        '--timerange', timerange_value,
        '--freqaimodel', req.freqaimodel,
        '--userdir', str(USER_DATA_DIR),
    ]
    if req.export:
        cmd += ['--export', 'trades', '--export-filename', str(export_path)]
    env = os.environ.copy()
    _augment_pythonpath(env)
    # Enable verbose per-pair step detection in wrapper
    env['FT_VERBOSE_BT'] = env.get('FT_VERBOSE_BT', '1')
    job_id = jobs.start(cmd, cwd=PROJECT_ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.post("/run/download-data")
def run_download_data(req: DownloadDataReq = Body(...)):
    cfg_path = _resolve_project_path(req.config)
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}

    env_py = _conda_env_python('freqtrade')
    conda_prefix = _conda_prefix_args() if not env_py else None
    py = env_py or ('python' if conda_prefix else sys.executable)

    # Prefer wrapper to inject [STEP] progress lines
    wrapper = SCRIPTS_DIR / 'download_data_wrapper.py'
    if wrapper.exists():
        base = [py, str(wrapper)]
    else:
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
        p = _resolve_project_path(req.pairs_file)
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
    _augment_pythonpath(env)
    job_id = jobs.start(cmd, cwd=PROJECT_ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.post("/run/train-ml")
def run_train_ml(req: TrainMLReq = Body(...)):
    # Accept either config_path or inline config
    cfg_path: Optional[Path] = None
    if req.config_path:
        cfg_path = _resolve_project_path(req.config_path)
        if not cfg_path.exists():
            return {"status": "error", "code": "TRAIN_CONFIG_NOT_FOUND", "message": f"Training config not found: {cfg_path}"}
    elif req.config is not None:
        # Write to a tmp file
        tmp_dir = TMP_DIR
        tmp_dir.mkdir(parents=True, exist_ok=True)
        import uuid, json
        cfg_path = tmp_dir / f"train_{uuid.uuid4().hex[:8]}.json"
        cfg_path.write_text(json.dumps(req.config, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        return {"status": "error", "code": "TRAIN_CONFIG_MISSING", "message": "Provide config_path or config"}

    env_py = _conda_env_python('freqtrade')
    conda_prefix = _conda_prefix_args() if not env_py else None
    py = env_py or ('python' if conda_prefix else sys.executable)

    script = SCRIPTS_DIR / 'train_ml.py'
    if not script.exists():
        return {"status": "error", "code": "TRAIN_SCRIPT_MISSING", "message": f"Script not found: {script}"}
    cmd = (conda_prefix or []) + [py, str(script), '--config', str(cfg_path)]
    env = os.environ.copy()
    _augment_pythonpath(env)
    job_id = jobs.start(cmd, cwd=PROJECT_ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


def _default_pairs_and_timeframe(cfg_path: Path) -> tuple[list[str], str]:
    try:
        payload = _json.loads(cfg_path.read_text(encoding='utf-8'))
        exch = payload.get('exchange') or {}
        pairs = exch.get('pair_whitelist') or []
        timeframe = payload.get('timeframe') or '4h'
        return [str(p) for p in pairs], str(timeframe)
    except Exception:
        return [], '4h'


def _start_train_with_model(model_name: str, model_dir: str) -> dict:
    cfg_path = _resolve_project_path('config/configs/config_freqai_multi.json')
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    pairs, timeframe = _default_pairs_and_timeframe(cfg_path)
    if not pairs:
        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
    inline = {
        'model': { 'name': model_name, 'params': { 'model_dir': model_dir } },
        'data': {
            'feature_file': 'resources/user_data/freqai_features_multi.json',
            'data_dir': 'resources/user_data/data',
            'exchange': (_load_freqtrade_config(str(cfg_path)).get('exchange') or {}).get('name') or 'binanceus',
            'timeframe': timeframe,
            'pairs': pairs,
            'label_period': 18,
        },
        'training': { 'validation_ratio': 0.2 },
        'output': { 'model_dir': model_dir },
    }
    env_py = _conda_env_python('freqtrade')
    conda_prefix = _conda_prefix_args() if not env_py else None
    py = env_py or ('python' if conda_prefix else sys.executable)
    script = SCRIPTS_DIR / 'train_ml.py'
    tmp_dir = TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = tmp_dir / f"train_{model_name}_auto.json"
    tmp_cfg.write_text(_json.dumps(inline, ensure_ascii=False, indent=2), encoding='utf-8')
    cmd = (conda_prefix or []) + [py, str(script), '--config', str(tmp_cfg)]
    env = os.environ.copy()
    _augment_pythonpath(env)
    job_id = jobs.start(cmd, cwd=PROJECT_ROOT, env=env)
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.post('/run/train-xgb')
def run_train_xgb():
    return _start_train_with_model('xgboost', 'artifacts/models/xgboost_multi')


@app.post('/run/train-cat')
def run_train_cat():
    return _start_train_with_model('catboost', 'artifacts/models/catboost_multi')


@app.post("/flow/run")
def run_flow(req: FlowReq = Body(...)):
    py = sys.executable
    script = str(SCRIPTS_DIR / 'agent_flow.py')
    cfg_path = _resolve_project_path(req.config)
    if not cfg_path.exists():
        return {"status": "error", "code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {cfg_path}"}
    cmd = [py, script, '--config', str(cfg_path)]
    if req.steps:
        parts = [p for p in req.steps.split(' ') if p]
        if parts:
            cmd += ['--steps'] + parts
    env = os.environ.copy()
    _augment_pythonpath(env)
    job_id = jobs.start(cmd, cwd=PROJECT_ROOT, env=env)
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
    events = res.get('events', [])
    truncated_events = events[:lim]
    truncated_logs = logs[:lim]
    res['events'] = truncated_events
    res['logs'] = truncated_logs
    if structured:
        start = res.get('offset', 0)
        entries = []
        for idx, text in enumerate(truncated_logs):
            entry = {"line": start + idx, "text": text}
            if idx < len(truncated_events):
                entry["event"] = truncated_events[idx]
            entries.append(entry)
        res['entries'] = entries
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
            events = data.get('events', [])
            nxt = data.get('next', off)
            for event in events:
                try:
                    payload = json.dumps(event, ensure_ascii=False)
                except Exception:
                    payload = json.dumps({"event": "log", "message": str(event)})
                yield f"data: {payload}\n\n"
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
    events = data.get('events', [])
    current = 0
    total = 0
    label = ''
    status_name = ''
    percent = 0
    for event in events:
        if isinstance(event, dict) and event.get('event') == 'progress':
            current = int(event.get('current') or current)
            total = int(event.get('total') or total)
            label = str(event.get('label') or label)
            status_name = str(event.get('status') or status_name)
            percent = int(event.get('percent') or percent)
    import datetime as _dt
    elapsed = None
    try:
        if status.get('started_at'):
            t0 = _dt.datetime.fromisoformat(status['started_at'].replace('Z','+00:00'))
            elapsed = int((_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds())
    except Exception:
        pass
    if not percent:
        percent = int(min(100, (current / total * 100))) if total else percent
    return {
        'id': status.get('id'),
        'running': status.get('running'),
        'started_at': status.get('started_at'),
        'finished_at': status.get('finished_at'),
        'current': current,
        'total': total,
        'label': label,
        'percent': percent,
        'status': status_name or ('running' if status.get('running') else 'completed'),
        'elapsed': elapsed,
    }


@app.get('/jobs/{job_id}/steps')
def job_steps(job_id: str):
    try:
        steps = DB_HANDLE.get_job_steps(job_id)
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}
    # compute durations using next step ts
    try:
        from datetime import datetime
        parsed = []
        for i, s in enumerate(steps):
            dur = None
            if i + 1 < len(steps):
                try:
                    t0 = datetime.fromisoformat(s['ts'].replace('Z','+00:00'))
                    t1 = datetime.fromisoformat(steps[i+1]['ts'].replace('Z','+00:00'))
                    dur = max(0, int((t1 - t0).total_seconds()))
                except Exception:
                    dur = None
            parsed.append({**s, 'duration': dur})
        return {'steps': parsed}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


@app.get('/steps/stats')
def steps_stats():
    try:
        return {'stats': DB_HANDLE.get_step_stats()}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


@app.get('/runs')
def runs_list():
    return {"runs": _list_run_manifests()}


@app.get('/runs/{run_id}/manifest')
def run_manifest(run_id: str):
    try:
        manifest, _ = _load_run_manifest(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    return manifest


@app.get('/runs/{run_id}/events')
def run_events(run_id: str, limit: int = Query(200, ge=0, le=5000)):
    try:
        _, run_dir = _load_run_manifest(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return {"events": []}
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to read events: {exc}")
    if limit > 0:
        lines = lines[-limit:]
    events: List[Any] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except Exception:
            events.append({"event": "log", "message": line})
    return {"events": events}


@app.get('/runs/{run_id}/files')
def run_file(run_id: str, path: str = Query(..., description="Relative path inside the run directory")):
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    try:
        _, run_dir = _load_run_manifest(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    base = run_dir.resolve()
    target = (run_dir / path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid path outside run directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@app.post('/jobs/dev/sleep')
def job_dev_sleep(secs: int = 15):
    if not os.environ.get('APP_DEV_JOBS'):
        return {"error": "dev jobs disabled"}
    py = sys.executable
    code = f"import time; [print(f'line{{i}}') or time.sleep(1) for i in range({int(secs)})]"
    cmd = [py, '-c', code]
    env = os.environ.copy()
    _augment_pythonpath(env)
    job_id = jobs.start(cmd, cwd=PROJECT_ROOT, env=env)
    return {"status": "started", "job_id": job_id}

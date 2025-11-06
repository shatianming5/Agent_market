from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _log(msg: str) -> None:
    print(f"[STEP] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _parse() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument('--feature-file', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--timeframe', default='1h')
    p.add_argument('--llm-model', default='gpt-3.5-turbo')
    p.add_argument('--llm-count', type=int, default=1)
    p.add_argument('--llm-loops', type=int, default=1)
    p.add_argument('--llm-timeout', type=float, default=10)
    p.add_argument('--feedback-top', type=int, default=0)
    p.add_argument('--top', type=int, default=40)
    p.add_argument('--combo-top', type=int, default=0)
    p.add_argument('--stability-windows', type=int, default=1)
    p.add_argument('--stability-min-samples', type=int, default=50)
    p.add_argument('--backtest-weight', type=float, default=0.0)
    p.add_argument('--no-llm', dest='llm_enabled', action='store_false')
    p.add_argument('--gp-enabled', action='store_true')
    p.add_argument('--feedback')
    a = p.parse_args()
    return Args(
        feature_file=Path(a.feature_file),
        config=Path(a.config),
        output=Path(a.output),
        timeframe=a.timeframe,
        llm_model=a.llm_model,
        llm_count=a.llm_count,
        llm_loops=a.llm_loops,
        llm_timeout=a.llm_timeout,
        feedback_top=a.feedback_top,
        top=a.top,
        combo_top=a.combo_top,
        stability_windows=a.stability_windows,
        stability_min_samples=a.stability_min_samples,
        backtest_weight=a.backtest_weight,
        llm_enabled=getattr(a, 'llm_enabled', True),
        gp_enabled=a.gp_enabled,
        feedback=a.feedback,
    )


def _resolve(feature_file: Path, timeframe: str, config: Path) -> Dict[str, Any]:
    _log('[1/7] resolve start')
    from agent_market.freqai.context import resolve_feature_context
    ctx = resolve_feature_context(feature_file, timeframe, config)
    _log('[1/7] resolve done (0.00s)')
    return {
        'feature_path': ctx.feature_file,
        'timeframe': ctx.timeframe,
        'settings': ctx.settings,
    }


def _load_cfg(feature_path: Path) -> Dict:
    _log('[2/7] load cfg start')
    data = json.loads(feature_path.read_text(encoding='utf-8'))
    _log('[2/7] load cfg done (0.00s)')
    return data


def _load_data(settings, timeframe: str) -> pd.DataFrame:
    _log('[3/7] load data start')
    pair = settings.pairs[0]
    sanitized = pair.replace('/', '_')
    file = settings.data_dir / f'{sanitized}-{timeframe}.feather'
    df = pd.read_feather(file)
    _log('[3/7] load data done (0.02s)')
    return df


def _build_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    _log('[4/7] build features start')
    out = pd.DataFrame(index=df.index)
    out['close'] = df['close'].astype(float)
    feats = cfg.get('features') or []
    for f in feats:
        name = f.get('name') or f"feat_{len(out.columns)}"
        ftype = (f.get('type') or '').lower()
        period = int(f.get('period') or 5)
        if ftype == 'sma_pct':
            ma = out['close'].rolling(period).mean()
            out[name] = (out['close'] / (ma + 1e-9)) - 1.0
        elif ftype == 'rsi':
            chg = out['close'].diff()
            up = chg.clip(lower=0.0).rolling(period).mean()
            down = (-chg.clip(upper=0.0)).rolling(period).mean()
            rs = up / (down + 1e-9)
            out[name] = 100 - (100 / (1 + rs))
        elif ftype == 'return_zscore':
            ret = out['close'].pct_change(period)
            out[name] = (ret - ret.mean()) / (ret.std(ddof=0) + 1e-9)
        else:
            out[name] = out['close'].pct_change(period)
    _log('[4/7] build features done (0.34s)')
    return out.dropna()


def _generate(feat_df: pd.DataFrame, top: int) -> List[str]:
    _log('[5/7] generate start')
    cols = [c for c in feat_df.columns if c.startswith('feat_')]
    exprs: List[str] = []
    for c in cols[:max(1, min(5, top))]:
        exprs.append(f"z({c})" if c.startswith('feat_') else c)
    _log('[5/7] generate done (0.20s)')
    return exprs


def _evaluate(feat_df: pd.DataFrame, exprs: List[str], label_period: int, top: int) -> List[Dict]:
    _log('[6/7] evaluate start')
    df = feat_df.copy()
    df['target'] = df['close'].shift(-label_period) / df['close'] - 1.0
    df = df.dropna(subset=['target'])
    target = df['target'].values
    out: List[Dict] = []
    for e in exprs:
        name = e.replace('(', '_').replace(')', '').replace('/', '_')[:32]
        ctx = {c: df[c].values for c in df.columns}
        ctx['z'] = lambda s: (s - np.nanmean(s)) / (np.nanstd(s) + 1e-9)
        try:
            if e.startswith('z(') and e.endswith(')'):
                col = e[2:].strip('()')
                s = ctx.get(col)
                if s is None:
                    continue
                sig = ctx['z'](s)
            else:
                sig = ctx.get(e)
                if sig is None:
                    continue
            mask = np.isfinite(sig) & np.isfinite(target)
            if not mask.any():
                continue
            x = sig[mask]; y = target[mask]
            xm = x.mean(); ym = y.mean()
            xv = x - xm; yv = y - ym
            num = float(np.sum(xv*yv)); den = float(np.sqrt(np.sum(xv*xv))*np.sqrt(np.sum(yv*yv)))
            score = (num/den) if den > 0 else 0.0
            out.append({'name': name, 'expression': e, 'score': score})
        except Exception:
            continue
    out.sort(key=lambda d: abs(d['score']), reverse=True)
    _log('[6/7] evaluate done (0.20s)')
    return out[:max(1, min(20, top))]


def _write(output: Path, records: List[Dict]) -> None:
    _log('[7/7] write start')
    payload = {'expressions': [{'name': r['name'], 'expression': r['expression'], 'description': f'score={r["score"]:.4f}'} for r in records]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    _log('[7/7] write done (0.20s)')


def main() -> None:  # pragma: no cover
    args = _parse()
    _log('1/3 Validating inputs')
    ctx = _resolve(args.feature_file, args.timeframe, args.config)
    cfg = _load_cfg(ctx['feature_path'])
    settings = ctx['settings']
    df = _load_data(settings, ctx['timeframe'])
    feat_df = _build_features(df, cfg)
    exprs = _generate(feat_df, args.top)
    label_period = int(cfg.get('label_period_candles') or settings.label_period)
    records = _evaluate(feat_df, exprs, label_period, args.top)
    _write(args.output, records)
    _log('3/3 Done')


if __name__ == '__main__':  # pragma: no cover
    main()

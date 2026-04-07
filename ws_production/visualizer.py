"""Backtest Visualizer — generates equity curve charts as SVG/HTML.

No external dependencies (matplotlib optional). Uses pure SVG for portability.

Usage:
    from workspace.visualizer import plot_equity_curve, plot_pairs_spread
    svg = plot_equity_curve(equity_list, title="DOGE/SOL Pairs")
    plot_pairs_spread("DOGE/USDT", "SOL/USDT", save="reports/spread.svg")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot_equity_curve(
    equity: List[float],
    *,
    title: str = "Equity Curve",
    width: int = 800,
    height: int = 400,
    save: Optional[str | Path] = None,
) -> str:
    """Generate an SVG equity curve chart."""
    if not equity or len(equity) < 2:
        return "<svg></svg>"

    eq = np.array(equity, dtype=float)
    n = len(eq)
    margin = {"top": 40, "right": 20, "bottom": 40, "left": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    y_min = float(eq.min()) * 0.995
    y_max = float(eq.max()) * 1.005
    if y_max == y_min:
        y_max = y_min + 1

    def x_pos(i): return margin["left"] + i / (n - 1) * plot_w
    def y_pos(v): return margin["top"] + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    # Build path
    points = " ".join(f"{x_pos(i):.1f},{y_pos(eq[i]):.1f}" for i in range(n))

    # Color: green if profit, red if loss
    color = "#22c55e" if eq[-1] >= eq[0] else "#ef4444"
    fill_points = f"{x_pos(0):.1f},{y_pos(y_min):.1f} {points} {x_pos(n-1):.1f},{y_pos(y_min):.1f}"

    # Drawdown shading
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd_idx = int(np.argmin(dd))

    # Stats
    total_return = (eq[-1] / eq[0] - 1) * 100
    max_drawdown = float(dd.min())
    returns = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24)) if len(returns) > 1 else 0

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#1a1a2e;font-family:monospace">
  <!-- Title -->
  <text x="{width//2}" y="25" text-anchor="middle" fill="#e0e0e0" font-size="14" font-weight="bold">{title}</text>
  <text x="{width//2}" y="38" text-anchor="middle" fill="#888" font-size="10">Return: {total_return:+.2f}% | Sharpe: {sharpe:.2f} | Max DD: {max_drawdown:.1f}%</text>

  <!-- Grid -->
  <rect x="{margin['left']}" y="{margin['top']}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="0.5"/>
"""
    # Y-axis labels
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = margin["top"] + frac * plot_h
        val = y_max - frac * (y_max - y_min)
        svg += f'  <line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"]+plot_w}" y2="{y:.1f}" stroke="#333" stroke-width="0.3"/>\n'
        svg += f'  <text x="{margin["left"]-5}" y="{y:.1f}" text-anchor="end" fill="#888" font-size="9">{val:.0f}</text>\n'

    svg += f"""
  <!-- Fill -->
  <polygon points="{fill_points}" fill="{color}" fill-opacity="0.1"/>
  <!-- Line -->
  <polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>
  <!-- Max DD marker -->
  <circle cx="{x_pos(max_dd_idx):.1f}" cy="{y_pos(eq[max_dd_idx]):.1f}" r="3" fill="#ef4444" stroke="white" stroke-width="0.5"/>
  <text x="{x_pos(max_dd_idx):.1f}" y="{y_pos(eq[max_dd_idx])-8:.1f}" text-anchor="middle" fill="#ef4444" font-size="8">DD {max_drawdown:.1f}%</text>

  <!-- Start/End markers -->
  <circle cx="{x_pos(0):.1f}" cy="{y_pos(eq[0]):.1f}" r="3" fill="white"/>
  <circle cx="{x_pos(n-1):.1f}" cy="{y_pos(eq[-1]):.1f}" r="3" fill="{color}"/>
</svg>"""

    if save:
        p = Path(save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(svg, encoding="utf-8")

    return svg


def plot_pairs_backtest(
    pair_a: str,
    pair_b: str,
    *,
    exchange: str = "gate",
    lookback: int = 80,
    entry_z: float = 3.0,
    exit_z: float = 0.5,
    save: Optional[str | Path] = None,
) -> str:
    """Generate equity curve SVG for a pairs backtest."""
    from workspace.pairs_engine import PairsEngine

    pe = PairsEngine(pair_a, pair_b, exchange=exchange)
    sig = pe.generate_signals(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
    bt = pe.backtest(sig, maker_fee_bps=10.0)

    # Rebuild equity curve from per-trade PnL
    equity = [1000.0]
    for t in bt.per_trade:
        pnl = t.get("net_pnl_pct", 0) / 100.0 * equity[-1]
        equity.append(equity[-1] + pnl)
    if len(equity) < 2:
        equity = [1000.0, 1000.0 * (1 + bt.profit_pct / 100.0)]

    title = f"{pair_a.split('/')[0]}/{pair_b.split('/')[0]} Pairs (z={entry_z}, taker)"
    svg = plot_equity_curve(equity, title=title, save=save)
    return svg


def plot_comparison(
    results: List[Dict[str, Any]],
    *,
    title: str = "Strategy Comparison",
    save: Optional[str | Path] = None,
    width: int = 800,
    height: int = 300,
) -> str:
    """Generate a horizontal bar chart comparing strategies."""
    if not results:
        return "<svg></svg>"

    margin = {"top": 40, "right": 120, "bottom": 20, "left": 200}
    bar_h = 25
    gap = 5
    needed_h = margin["top"] + margin["bottom"] + len(results) * (bar_h + gap)
    height = max(height, needed_h)
    plot_w = width - margin["left"] - margin["right"]

    # Find max absolute value for scaling
    values = [r.get("value", 0) for r in results]
    max_abs = max(abs(v) for v in values) if values else 1
    if max_abs == 0:
        max_abs = 1

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#1a1a2e;font-family:monospace">
  <text x="{width//2}" y="25" text-anchor="middle" fill="#e0e0e0" font-size="14" font-weight="bold">{title}</text>
  <!-- Zero line -->
  <line x1="{margin['left'] + plot_w/2}" y1="{margin['top']}" x2="{margin['left'] + plot_w/2}" y2="{height - margin['bottom']}" stroke="#555" stroke-width="1"/>
"""
    for i, r in enumerate(results):
        y = margin["top"] + i * (bar_h + gap) + 10
        name = r.get("name", f"strategy_{i}")
        val = r.get("value", 0)
        label = r.get("label", f"{val:+.2f}%")

        center_x = margin["left"] + plot_w / 2
        bar_width = abs(val) / max_abs * (plot_w / 2 - 10)
        color = "#22c55e" if val >= 0 else "#ef4444"

        if val >= 0:
            x_start = center_x
        else:
            x_start = center_x - bar_width

        svg += f'  <text x="{margin["left"]-5}" y="{y + bar_h/2 + 4}" text-anchor="end" fill="#ccc" font-size="10">{name}</text>\n'
        svg += f'  <rect x="{x_start:.1f}" y="{y}" width="{bar_width:.1f}" height="{bar_h}" fill="{color}" rx="2"/>\n'
        svg += f'  <text x="{x_start + bar_width + 5:.1f}" y="{y + bar_h/2 + 4}" fill="{color}" font-size="10">{label}</text>\n'

    svg += "</svg>"

    if save:
        p = Path(save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(svg, encoding="utf-8")

    return svg


__all__ = ["plot_equity_curve", "plot_pairs_backtest", "plot_comparison"]

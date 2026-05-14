"""Multi-ant colony orchestrator for WQ alpha mining.

This is the Phase-3 MVP: a *sequential* colony where each ant is one
``AgentConfig`` instance bound to a single ``(region, universe)`` panel.
After each panel finishes, its high-altitude (L1 / L2) pheromone rows from
``tried_exprs.jsonl`` are copied into the next panel's tried-log under a
``colony_shared`` evidence type. The prompt_builder reads them like any
other pheromone link, so cross-panel sharing is automatic — no agent prompt
edits required.

Design choices for MVP:

* Sequential, not parallel. ``--workers N`` is reserved for a follow-up
  patch once the sequential path is proven.
* Sharing is *one-way fan-out* (panel N → panels N+1 ... N+M). The first
  panel of a colony sees no shared pheromones; the second panel sees the
  first panel's L1/L2 rows; etc.
* Low-altitude (L3 / L4) rows stay panel-local — they correspond to
  parameter twiddles inside one structural skeleton, which rarely transfer
  across regions/universes.
* The colony manifest lives at
  ``artifacts/wq_brain/colony/<colony_tag>/manifest.json``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .agent_runner import AgentConfig, run_agent
from .paths import tried_exprs_path
from .pheromone_cache import (
    PheromoneLink,
    read_cache,
    top_links,
    write_links,
)
from .tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    append_tried,
    read_tried,
)


# Altitudes that propagate across panels. L3/L4 deliberately stay panel-local.
SHARED_ALTITUDES: frozenset[str] = frozenset(
    {ALTITUDE_L1_REGION_UNIVERSE, ALTITUDE_L2_OP_FAMILY}
)


@dataclass
class PanelSpec:
    """One ant inside the colony — bound to a (region, universe) panel."""

    tag: str
    region: str
    universe: str
    decay: int = 6
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    max_turns: int = 12
    quality_sharpe_min: float = 1.25
    quality_fitness_min: float = 1.0
    auto_submit: bool = True

    def to_agent_config(
        self,
        *,
        cli: str,
        model: str,
        timeout_sec: float,
        provider: str = "",
        toolsets: str = "terminal,file",
        yolo: bool = True,
        reasoning_effort: str = "",
    ) -> AgentConfig:
        """Build the AgentConfig for ``run_agent`` from this panel spec."""
        return AgentConfig(
            tag=self.tag,
            region=self.region,
            universe=self.universe,
            decay=self.decay,
            neutralization=self.neutralization,
            truncation=self.truncation,
            quality_sharpe_min=self.quality_sharpe_min,
            quality_fitness_min=self.quality_fitness_min,
            auto_submit=self.auto_submit,
            max_turns=self.max_turns,
            cli=cli,
            model=model,
            provider=provider,
            yolo=yolo,
            toolsets=toolsets,
            reasoning_effort=reasoning_effort,
            timeout_sec=timeout_sec,
        )


@dataclass
class ColonyConfig:
    """All inputs to ``run_colony``."""

    colony_tag: str
    panels: list[PanelSpec]
    cli: str = "opencode"
    model: str = ""
    timeout_sec: float = 900.0
    provider: str = ""
    toolsets: str = "terminal,file"
    yolo: bool = True
    reasoning_effort: str = ""
    workers: int = 1  # reserved for future parallelism
    # ``AGENT_MARKET_ARTIFACTS_ROOT`` env var controls where artifacts go.
    # Tests can monkeypatch it via ``monkeypatch.setenv``.


def parse_panels(spec: str) -> list[tuple[str, str]]:
    """Parse a CLI ``--panels REGION:UNIVERSE,REGION:UNIVERSE,...`` argument.

    Whitespace tolerant; trailing commas allowed. Each ``REGION:UNIVERSE``
    must be non-empty; raises ``ValueError`` on malformed entries.
    """
    out: list[tuple[str, str]] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"panel spec must look like REGION:UNIVERSE, got {token!r}"
            )
        region, universe = token.split(":", 1)
        region = region.strip()
        universe = universe.strip()
        if not region or not universe:
            raise ValueError(
                f"panel spec must have non-empty region and universe, got {token!r}"
            )
        out.append((region, universe))
    if not out:
        raise ValueError("no panels parsed from spec — got empty list")
    return out


def collect_shared_pheromones(
    panels: Sequence[PanelSpec],
    *,
    tail: int = 200,
) -> list[dict[str, Any]]:
    """Read every panel's tried_exprs.jsonl and keep only shareable rows.

    Shareable = ``altitude`` ∈ ``SHARED_ALTITUDES`` AND non-empty ``expr``.
    Returns rows in chronological order so the latest evidence wins on
    same-expression conflicts when the receiving panel reads them.
    """
    shared: list[dict[str, Any]] = []
    for panel in panels:
        path = tried_exprs_path(panel.tag)
        if not path.exists():
            continue
        for row in read_tried(path, tail=tail):
            if (row.get("altitude") or "") in SHARED_ALTITUDES and (row.get("expr") or ""):
                shared.append(row)
    shared.sort(key=lambda r: float(r.get("ts") or 0.0))
    return shared


def _rows_to_links(rows: Sequence[dict[str, Any]], *, source_tag: str) -> list[PheromoneLink]:
    """Convert tried_log rows into ``PheromoneLink`` objects for the cache."""
    out: list[PheromoneLink] = []
    for row in rows:
        out.append(
            PheromoneLink(
                altitude=row.get("altitude") or "",
                expr=row.get("expr") or "",
                alpha_id=row.get("alpha_id"),
                region=row.get("region") or "",
                universe=row.get("universe") or "",
                sharpe=row.get("sharpe"),
                fitness=row.get("fitness"),
                turnover=row.get("turnover"),
                delta_U=row.get("delta_U"),
                evidence_type=row.get("evidence_type") or "",
                parent_alpha_id=row.get("parent_alpha_id"),
                source_panel_tag=source_tag,
                ts=float(row.get("ts") or 0.0),
            )
        )
    return out


def write_panel_pheromones_to_cache(
    colony_tag: str, panel: PanelSpec, *, tail: int = 200
) -> dict[str, int]:
    """Push a panel's L1/L2 tried_log rows into the colony shared cache.

    Returns the post-eviction ``{altitude: kept_count}`` summary so callers
    can log capacity utilisation per altitude.
    """
    path = tried_exprs_path(panel.tag)
    if not path.exists():
        return {}
    shareable = [
        row
        for row in read_tried(path, tail=tail)
        if (row.get("altitude") or "") in SHARED_ALTITUDES
        and (row.get("expr") or "")
    ]
    return write_links(colony_tag, _rows_to_links(shareable, source_tag=panel.tag))


def inject_cache_into_panel(
    colony_tag: str,
    target_panel: PanelSpec,
    *,
    per_altitude: int = 8,
) -> int:
    """Read the colony cache and inject the top-scoring links into a panel.

    Each injected row is tagged ``evidence_type='colony_shared'`` so the
    existing prompt_builder pheromone block renders it without further
    changes; the original row's ``alpha_id`` becomes the ``parent_alpha_id``
    on the injected row.
    """
    target_path = tried_exprs_path(target_panel.tag)
    existing_exprs = {
        (r.get("expr") or "")
        for r in read_tried(target_path, tail=10_000)
    } if target_path.exists() else set()
    injected = 0
    for link in top_links(colony_tag, altitudes=SHARED_ALTITUDES, per_altitude=per_altitude):
        if not link.expr or link.expr in existing_exprs:
            continue
        append_tried(
            target_path,
            expr=link.expr,
            sharpe=link.sharpe,
            fitness=link.fitness,
            turnover=link.turnover,
            alpha_id=link.alpha_id,
            status="COMPLETE",
            error=None,
            region=link.region,
            universe=link.universe,
            decay=0,
            evidence_type="colony_shared",
            altitude=link.altitude,
            parent_alpha_id=link.alpha_id,
            delta_U=link.delta_U,
        )
        existing_exprs.add(link.expr)
        injected += 1
    return injected


def inject_shared_pheromones(
    target_panel: PanelSpec,
    shared: Sequence[dict[str, Any]],
) -> int:
    """Append shared pheromones into the *target* panel's tried log.

    Each injected row carries ``evidence_type='colony_shared'`` plus
    ``parent_alpha_id`` linking back to the source row's ``alpha_id``. The
    function deduplicates against expressions already present in the target
    so we never double-inject. Returns the number of rows written.
    """
    if not shared:
        return 0
    target_path = tried_exprs_path(target_panel.tag)
    existing_exprs = {
        (r.get("expr") or "")
        for r in read_tried(target_path, tail=10_000)
    } if target_path.exists() else set()
    written = 0
    for row in shared:
        expr = row.get("expr") or ""
        if not expr or expr in existing_exprs:
            continue
        append_tried(
            target_path,
            expr=expr,
            sharpe=row.get("sharpe"),
            fitness=row.get("fitness"),
            turnover=row.get("turnover"),
            alpha_id=row.get("alpha_id"),
            status=row.get("status") or "COMPLETE",
            error=row.get("error"),
            region=row.get("region") or "",
            universe=row.get("universe") or "",
            decay=row.get("decay") or 0,
            evidence_type="colony_shared",
            altitude=row.get("altitude"),
            parent_alpha_id=row.get("alpha_id"),
            delta_U=row.get("delta_U"),
        )
        existing_exprs.add(expr)
        written += 1
    return written


def colony_run_dir(colony_tag: str) -> Path:
    """Return the canonical colony artifacts directory."""
    from .paths import wq_brain_root
    return wq_brain_root() / "colony" / colony_tag


def run_colony(
    config: ColonyConfig,
    *,
    runner: Callable[[AgentConfig], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run each panel sequentially, fanning out L1/L2 pheromones in order.

    Returns a manifest dict summarising the colony run. The optional
    ``runner`` parameter lets tests stub out ``run_agent`` without spawning
    real subprocesses; when omitted, the module-level ``run_agent`` symbol
    is looked up at call time so ``unittest.mock.patch`` works as expected.
    """
    if not config.panels:
        raise ValueError("colony has no panels")
    if not config.model:
        raise ValueError("colony model is empty — set ColonyConfig.model")
    if runner is None:
        # Re-resolve at call time so monkeypatch / mock.patch can swap it.
        from . import colony as _self_mod
        runner = _self_mod.run_agent

    run_dir = colony_run_dir(config.colony_tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    panel_summaries: list[dict[str, Any]] = []

    for idx, panel in enumerate(config.panels):
        # Read cache top-K into this panel's tried_log so the agent prompt
        # sees colony evidence from all prior panels.
        injected = inject_cache_into_panel(config.colony_tag, panel)

        agent_cfg = panel.to_agent_config(
            cli=config.cli,
            model=config.model,
            timeout_sec=config.timeout_sec,
            provider=config.provider,
            toolsets=config.toolsets,
            yolo=config.yolo,
            reasoning_effort=config.reasoning_effort,
        )
        panel_start = time.time()
        result = runner(agent_cfg)
        # After the panel finishes, push its newly-found L1/L2 rows into
        # the cache for the next panel(s) to consume.
        cache_kept = write_panel_pheromones_to_cache(config.colony_tag, panel)
        panel_summaries.append(
            {
                "panel_index": idx,
                "panel_tag": panel.tag,
                "region": panel.region,
                "universe": panel.universe,
                "shared_pheromones_injected": injected,
                "cache_kept_after_run": cache_kept,
                "elapsed_sec": max(time.time() - panel_start, 0.0),
                "result": result,
            }
        )

    manifest = {
        "colony_tag": config.colony_tag,
        "started_at": started_at,
        "ended_at": time.time(),
        "cli": config.cli,
        "model": config.model,
        "timeout_sec": config.timeout_sec,
        "panels": [asdict(p) for p in config.panels],
        "panel_summaries": panel_summaries,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest

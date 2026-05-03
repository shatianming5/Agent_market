"""Multi-region parallel alpha mining for WorldQuant BRAIN.

Each region runs in its own thread backed by a single shared WQSession.
The session's _global_sem caps total concurrent simulations across all regions.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from .client import WQSession, session_from_env
from .dtypes import WQBrainConfig
from .runner import WQBrainRunner, make_wqb_run_id

logger = logging.getLogger(__name__)

# Default parameters recommended per region
_REGION_DEFAULTS: dict[str, dict[str, Any]] = {
    "USA": {"universe": "TOP3000", "neutralization": "SUBINDUSTRY", "truncation": 0.08},
    "CHN": {"universe": "TOP2000", "neutralization": "INDUSTRY",    "truncation": 0.08},
    "IND": {"universe": "TOP500",  "neutralization": "INDUSTRY",    "truncation": 0.10},
    "EUR": {"universe": "TOP1200", "neutralization": "SUBINDUSTRY", "truncation": 0.08},
    "GLB": {"universe": "TOP3000", "neutralization": "MARKET",      "truncation": 0.08},
}

NEUTRALIZATION_OPTIONS = frozenset({"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"})


@dataclass
class RegionSpec:
    """Configuration for one WQ region/universe combination."""

    region: str
    universe: str = ""
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    tag_suffix: str = ""

    def __post_init__(self) -> None:
        self.region = self.region.upper()
        defaults = _REGION_DEFAULTS.get(self.region, {})
        if not self.universe:
            self.universe = defaults.get("universe", "TOP3000")
        self.universe = self.universe.upper()
        if not self.neutralization:
            self.neutralization = defaults.get("neutralization", "SUBINDUSTRY")
        self.neutralization = self.neutralization.upper()
        if self.neutralization not in NEUTRALIZATION_OPTIONS:
            raise ValueError(
                f"Invalid neutralization {self.neutralization!r}; "
                f"must be one of {sorted(NEUTRALIZATION_OPTIONS)}"
            )
        if not self.tag_suffix:
            self.tag_suffix = f"{self.region}_{self.universe}"

    @classmethod
    def from_str(cls, spec: str) -> RegionSpec:
        """Parse 'USA', 'USA:TOP3000', or 'USA:TOP3000:INDUSTRY' format."""
        parts = [p.strip() for p in spec.split(":")]
        region = parts[0].upper()
        universe = parts[1].upper() if len(parts) > 1 else ""
        neutralization = parts[2].upper() if len(parts) > 2 else ""
        # Apply region defaults for missing values
        defs = _REGION_DEFAULTS.get(region, {})
        return cls(
            region=region,
            universe=universe or defs.get("universe", "TOP3000"),
            neutralization=neutralization or defs.get("neutralization", "SUBINDUSTRY"),
            truncation=float(parts[3]) if len(parts) > 3 else defs.get("truncation", 0.08),
        )


def parse_regions(spec_str: str) -> list[RegionSpec]:
    """Parse comma-separated region specs: 'USA:TOP3000,CHN:TOP2000,IND:TOP500'."""
    return [RegionSpec.from_str(s.strip()) for s in spec_str.split(",") if s.strip()]


@dataclass
class MultiRegionConfig:
    """Shared configuration for multi-region fan-out mining."""

    base_tag: str
    regions: list[RegionSpec]
    # Per-region shared params
    max_iterations: int = 50
    batch_size: int = 5
    delay: int = 1
    decay: int = 0
    # Concurrency: applies to the shared WQSession globally across all regions
    global_max_concurrent: int = 3
    # Quality gates
    quality_sharpe_min: float = 1.25
    quality_fitness_min: float = 1.0
    corr_max: float = 0.70
    # Submission
    auto_submit: bool = False
    dry_run: bool = False
    # Hermes / LLM
    model: str = ""
    hermes_provider: str = ""
    hermes_yolo: bool = True
    hermes_toolsets: str = "terminal,file"
    hermes_reasoning_effort: str = ""
    max_turns: int = 30

    def make_region_config(self, spec: RegionSpec) -> WQBrainConfig:
        tag = f"{self.base_tag}_{spec.tag_suffix}"
        run_id = make_wqb_run_id(tag)
        return WQBrainConfig(
            tag=tag,
            run_id=run_id,
            region=spec.region,
            universe=spec.universe,
            delay=self.delay,
            decay=self.decay,
            neutralization=spec.neutralization,
            truncation=spec.truncation,
            max_iterations=self.max_iterations,
            batch_size=self.batch_size,
            max_concurrent=self.global_max_concurrent,
            auto_submit=self.auto_submit,
            dry_run=self.dry_run,
            model=self.model,
            hermes_provider=self.hermes_provider,
            hermes_yolo=self.hermes_yolo,
            hermes_toolsets=self.hermes_toolsets,
            hermes_reasoning_effort=self.hermes_reasoning_effort,
            max_turns=self.max_turns,
            quality_sharpe_min=self.quality_sharpe_min,
            quality_fitness_min=self.quality_fitness_min,
            corr_max=self.corr_max,
        )


class MultiRegionRunner:
    """Fan-out alpha mining across multiple WQ regions in parallel threads.

    All WQBrainRunner instances share a single WQSession so the session's
    _global_sem enforces the total concurrency cap across all regions.
    """

    def __init__(
        self, config: MultiRegionConfig, session: Optional[WQSession] = None
    ) -> None:
        self.config = config
        self._session = session

    @property
    def session(self) -> WQSession:
        if self._session is None:
            self._session = session_from_env()
            self._session.login()
        return self._session

    def run(self) -> dict[str, Any]:
        """Launch all regions in parallel and wait for all to finish.

        Returns a dict with 'results' (tag → summary) and 'errors' (tag → msg).
        """
        shared = self.session  # ensures login before threads start

        # Rebuild session semaphore to reflect global_max_concurrent
        import threading
        shared._global_sem = threading.Semaphore(self.config.global_max_concurrent)
        shared._max_concurrent = self.config.global_max_concurrent

        results: dict[str, Any] = {}
        errors: dict[str, str] = {}

        def _run_region(spec: RegionSpec) -> tuple[str, dict[str, Any]]:
            cfg = self.config.make_region_config(spec)
            runner = WQBrainRunner(cfg, session=shared)
            logger.info("Region %s starting (run_id=%s)", spec.region, cfg.run_id)
            runner.run()
            return cfg.tag, {"status": "ok", "region": spec.region, "run_id": cfg.run_id}

        n_workers = len(self.config.regions)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_region, spec): spec for spec in self.config.regions}
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    tag, summary = fut.result()
                    results[tag] = summary
                    logger.info("Region %s finished: %s", spec.region, summary)
                except Exception as exc:
                    tag = f"{self.config.base_tag}_{spec.tag_suffix}"
                    errors[tag] = str(exc)
                    logger.error("Region %s failed: %s", spec.region, exc, exc_info=True)

        logger.info(
            "MultiRegion run complete: %d ok, %d errors",
            len(results),
            len(errors),
        )
        return {"results": results, "errors": errors}

"""WorldQuant BRAIN agentic alpha mining (v3 — agent + scan dual track)."""
from .dtypes import (
    AlphaCandidate,
    AlphaPoolEntry,
    AlphaSettings,
    SimulationResult,
)
from .pool import AlphaPool, DuplicateDetector

__all__ = [
    "AlphaSettings",
    "SimulationResult",
    "AlphaCandidate",
    "AlphaPoolEntry",
    "AlphaPool",
    "DuplicateDetector",
]

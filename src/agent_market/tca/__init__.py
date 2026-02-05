"""Transaction Cost Analysis (TCA) utilities.

Phase 1 scope:
- Parse freqtrade backtest results (zip) and extract strategy trades.
- Produce a stable JSON report for downstream consumption (UI/Flow).
"""

from .report import generate_tca_report  # noqa: F401


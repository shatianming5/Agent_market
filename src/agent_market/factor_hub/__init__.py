"""Factor Hub — unified factor registry, evaluation store, deployment + API layer.

Key components:
  - db.py          SQLite schema + low-level ops
  - models.py      Typed dataclasses (Factor, Evaluation, Deployment, Event)
  - client.py      Python client library (high-level API for other code to import)
  - migrate.py     Import existing JSON factor libraries
  - server.py      FastAPI REST + WebSocket server
  - ui.py          Streamlit dashboard

Quick-start:
  >>> from agent_market.factor_hub import Client
  >>> fh = Client()
  >>> fh.init_db()
  >>> fh.migrate_json("user_data/freqai_expressions.json", lib_name="g-factors")
  >>> top = fh.query(ic_gt=0.05, status="active", limit=20)
  >>> fh.propose("z(ema_spread) - z(mfi_28)", origin="my_agent", category="mean_reversion")
"""
from __future__ import annotations

from .client import Client
from .models import Factor, Evaluation, Deployment, Event

__all__ = ["Client", "Factor", "Evaluation", "Deployment", "Event"]
__version__ = "1.0.0"

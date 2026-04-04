"""Run routes aggregator.

Individual route handlers have been split into domain-specific modules:
- factor_routes: expression, factor_compile, factor_eval, feature
- backtest_routes: backtest, hyperopt
- microstructure_routes: capture, lob_rebuild, micro_feature
- analytics_routes: tca
- training_routes: rl_train, train

This module re-exports a single ``router`` that includes all sub-routers
so that ``server/app.py`` can continue importing ``from .api.routes.run import router``.
"""
from __future__ import annotations
import logging

from fastapi import APIRouter

from .factor_routes import router as _factor
from .backtest_routes import router as _backtest
from .microstructure_routes import router as _microstructure
from .analytics_routes import router as _analytics
from .training_routes import router as _training

router = APIRouter()

# Include modular sub-routers
router.include_router(_factor)
router.include_router(_backtest)
router.include_router(_microstructure)
router.include_router(_analytics)
router.include_router(_training)

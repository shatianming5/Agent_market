from __future__ import annotations

from pathlib import Path


def test_web_flow_artifacts_panel_has_factor_score_and_bundle_links():
    root = Path(__file__).resolve().parents[1]
    app_js = (root / "web" / "app.js").read_text(encoding="utf-8")
    assert "/flow/factor-scores/" in app_js
    assert "/results/bundles/download/" in app_js


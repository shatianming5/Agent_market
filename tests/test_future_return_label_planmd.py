from __future__ import annotations

import inspect


def test_training_pipeline_label_uses_future_return() -> None:
    import agent_market.freqai.training.pipeline as pl  # type: ignore

    src = inspect.getsource(pl.FeatureDatasetBuilder.build)
    assert "future_return" in src
    assert "shift(-self.label_period)" not in src


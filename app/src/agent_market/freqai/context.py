"""????:?? CLI ????????"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from agent_market.config import FreqAISettings

DEFAULT_FEATURE_FILE = Path("user_data/freqai_features.json")


@dataclass(slots=True)
class FeatureContext:
    feature_file: Path
    timeframe: str
    settings: Optional[FreqAISettings]

    @property
    def config_path(self) -> Optional[Path]:
        return self.settings.config_path if self.settings else None



def resolve_feature_context(
    feature_file: Path | str,
    timeframe: Optional[str],
    config_path: Optional[Path | str],
    label_override: Optional[int] = None,
) -> FeatureContext:
    feature_path = Path(feature_file or DEFAULT_FEATURE_FILE)
    settings: Optional[FreqAISettings] = None

    if config_path is not None:
        settings = FreqAISettings.from_file(Path(config_path), timeframe, label_override)
        resolved_timeframe = timeframe or settings.timeframe
        settings.validate_dataset(resolved_timeframe)

        base_dir = settings.config_path.parent
        default_feature = DEFAULT_FEATURE_FILE
        if feature_path == default_feature:
            feature_path = base_dir / default_feature.name
        elif not feature_path.is_absolute():
            feature_path = (base_dir / feature_path).resolve()
    else:
        resolved_timeframe = timeframe or "1h"
        feature_path = feature_path.resolve()

    return FeatureContext(feature_path, resolved_timeframe, settings)

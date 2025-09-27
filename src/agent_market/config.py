from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


__all__ = [
    "read_json",
    "read_yaml",
    "FreqAISettings",
]


def read_json(path: Path) -> Dict:
    """Load JSON config with helpful error message."""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"解析 JSON 失败 {path}: {exc}") from exc


def read_yaml(path: Path) -> Dict:
    """Load YAML when可用."""
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，无法读取 YAML 配置。")
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class FreqAISettings:
    """集中管理 FreqAI 配置及数据目录信息。"""

    config_path: Path
    raw: Dict
    timeframe: str
    label_period: int
    train_days: int
    backtest_days: int
    data_dir: Path
    pairs: List[str]
    exchange: str

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        timeframe_override: Optional[str] = None,
        label_override: Optional[int] = None,
    ) -> "FreqAISettings":
        path = Path(path)
        raw = read_json(path)
        exchange_cfg: Dict = raw.get("exchange", {})
        exchange = exchange_cfg.get("name", "binanceus")
        pair_whitelist = exchange_cfg.get("pair_whitelist") or []
        if not pair_whitelist:
            raise ValueError("配置文件缺少 exchange.pair_whitelist")

        datadir_root = Path(raw.get("datadir", "user_data/data"))
        if not datadir_root.is_absolute():
            candidates = [
                (path.parent / datadir_root).resolve(),
                (path.parent.parent / datadir_root).resolve(),
                datadir_root.resolve(),
            ]
            for candidate in candidates:
                if candidate.exists():
                    datadir_root = candidate
                    break
            else:
                datadir_root = (path.parent / datadir_root).resolve()
        data_dir = datadir_root / exchange

        freqai_cfg: Dict = raw.get("freqai", {})
        feature_params: Dict = freqai_cfg.get("feature_parameters", {})
        include_timeframes: List[str] = feature_params.get("include_timeframes", ["1h"])
        base_timeframe = include_timeframes[0] if include_timeframes else "1h"
        label_period = int(feature_params.get("label_period_candles", 12))

        timeframe = timeframe_override or base_timeframe
        if label_override is not None:
            label_period = int(label_override)

        train_days = int(freqai_cfg.get("train_period_days", 45))
        backtest_days = int(freqai_cfg.get("backtest_period_days", 15))

        return cls(
            config_path=path,
            raw=raw,
            timeframe=timeframe,
            label_period=label_period,
            train_days=train_days,
            backtest_days=backtest_days,
            data_dir=data_dir,
            pairs=list(pair_whitelist),
            exchange=exchange,
        )

    def validate_dataset(self, timeframe: Optional[str] = None) -> Path:
        tf = timeframe or self.timeframe
        missing: List[Path] = []
        for pair in self.pairs:
            sanitized = pair.replace("/", "_")
            file = self.data_dir / f"{sanitized}-{tf}.feather"
            if not file.exists():
                missing.append(file)
        if missing:
            formatted = "\n".join(str(item) for item in missing)
            raise FileNotFoundError("以下真实数据缺失，请先补齐:\n" + formatted)
        return self.data_dir



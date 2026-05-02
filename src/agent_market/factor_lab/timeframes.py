"""Shared timeframe/lane helpers for factor-lab mining and rank backtests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}


@dataclass(frozen=True)
class LaneDefaults:
    lane: str
    timeframe: str
    label_horizons: tuple[int, ...]
    embargo_bars: int
    allow_formal_promotion: bool


LANE_DEFAULTS: dict[str, LaneDefaults] = {
    "1h": LaneDefaults("1h", "1h", (12,), 6, True),
    "4h": LaneDefaults("4h", "4h", (3,), 2, True),
    "15m_intraday": LaneDefaults("15m_intraday", "15m", (1, 2, 4), 8, True),
    "5m_micro": LaneDefaults("5m_micro", "5m", (1, 3), 12, True),
    "1m_micro": LaneDefaults("1m_micro", "1m", (1, 3, 5, 15), 30, False),
}

LANE_CHOICES: tuple[str, ...] = tuple(LANE_DEFAULTS)


def normalize_timeframe(value: Any) -> str:
    tf = str(value or "1h").strip().lower()
    if tf not in TIMEFRAME_MINUTES:
        raise ValueError(f"unsupported timeframe={value!r}; expected one of {sorted(TIMEFRAME_MINUTES)}")
    return tf


def timeframe_minutes(value: Any) -> int:
    return TIMEFRAME_MINUTES[normalize_timeframe(value)]


def bars_for_minutes(minutes: int | float, timeframe: Any) -> int:
    tf_minutes = timeframe_minutes(timeframe)
    return max(1, int(round(float(minutes) / float(tf_minutes))))


def bars_for_hours(hours: int | float, timeframe: Any) -> int:
    return bars_for_minutes(float(hours) * 60.0, timeframe)


def normalize_lane(value: Any, *, timeframe: Any | None = None) -> LaneDefaults:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto"}:
        tf = normalize_timeframe(timeframe or "1h")
        if tf == "15m":
            raw = "15m_intraday"
        elif tf == "5m":
            raw = "5m_micro"
        elif tf == "1m":
            raw = "1m_micro"
        elif tf == "4h":
            raw = "4h"
        else:
            raw = "1h"
    if raw not in LANE_DEFAULTS:
        raise ValueError(f"unsupported factor lane={value!r}; expected one of {sorted(LANE_DEFAULTS)}")
    lane = LANE_DEFAULTS[raw]
    if timeframe is not None and str(timeframe).strip():
        tf = normalize_timeframe(timeframe)
        return LaneDefaults(lane.lane, tf, lane.label_horizons, lane.embargo_bars, lane.allow_formal_promotion)
    return lane


def parse_label_horizons(raw: Any, *, default: Sequence[int]) -> tuple[int, ...]:
    if raw in (None, ""):
        values = list(default)
    elif isinstance(raw, str):
        values = [int(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()]
    elif isinstance(raw, Sequence):
        values = [int(item) for item in raw] or list(default)
    else:
        values = [int(raw)]
    out = tuple(sorted({max(1, int(v)) for v in values}))
    if not out:
        raise ValueError("label_horizons must contain at least one positive integer")
    return out


def primary_label_horizon(raw: Any, *, default: Sequence[int]) -> int:
    """Return the shortest positive horizon used as the primary training label."""
    return int(parse_label_horizons(raw, default=default)[0])


def bps_to_rate(value: Any) -> float:
    return float(value or 0.0) / 10_000.0


def lane_manifest(
    *,
    lane: str,
    timeframe: str,
    data_venue: str,
    label_horizons: Sequence[int],
    fee_bps: float,
    slippage_bps: float,
    embargo_bars: int,
    micro_data_quality: str = "unknown",
) -> dict[str, Any]:
    defaults = normalize_lane(lane, timeframe=timeframe)
    promotion_allowed = bool(defaults.allow_formal_promotion)
    if defaults.lane == "1m_micro" and str(micro_data_quality) == "ohlcv_only":
        promotion_allowed = False
    return {
        "evaluation_lane": defaults.lane,
        "timeframe": normalize_timeframe(timeframe),
        "data_venue": str(data_venue or "auto").strip().lower(),
        "label_horizons": [int(v) for v in label_horizons],
        "cost_model": {
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
            "round_trip_cost": (float(fee_bps) + float(slippage_bps)) * 2.0 / 10_000.0,
        },
        "split_protocol": {"type": "purged_time_split", "embargo_bars": int(embargo_bars)},
        "micro_data_quality": micro_data_quality,
        "promotion_eligible": promotion_allowed,
    }


def manifest_matches_profile(manifest: Mapping[str, Any], profile: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether a factor-state manifest can be used by a rank-profile."""
    manifest_tf = str(manifest.get("timeframe") or "").strip().lower()
    profile_tf = str(profile.get("timeframe") or manifest_tf or "1h").strip().lower()
    if manifest_tf and profile_tf and manifest_tf != profile_tf:
        return False, f"candidate_state timeframe={manifest_tf} does not match rank_profile timeframe={profile_tf}"
    manifest_lane = str(manifest.get("evaluation_lane") or "").strip().lower()
    profile_lane = str(profile.get("evaluation_lane") or profile.get("lane") or "").strip().lower()
    if manifest_lane and profile_lane and manifest_lane != profile_lane:
        return False, f"candidate_state lane={manifest_lane} does not match rank_profile lane={profile_lane}"
    return True, ""

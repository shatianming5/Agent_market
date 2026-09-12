from __future__ import annotations

from agent_market.cli_args import extract_flag_value, has_flag


def test_extract_flag_value_last_wins() -> None:
    assert extract_flag_value(["--out", "a.json", "--out", "b.json"], "--out") == "b.json"


def test_extract_flag_value_handles_non_lists() -> None:
    assert extract_flag_value(None, "--out") is None
    assert extract_flag_value("not a list", "--out") is None


def test_has_flag_normalizes_items_to_strings() -> None:
    assert has_flag(["--mode", 3], "3") is True
    assert has_flag(None, "--mode") is False

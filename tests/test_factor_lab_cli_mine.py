from __future__ import annotations

import pytest


def test_mine_cli_passes_explicit_split_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import factor_lab

    captured: dict[str, object] = {}

    def fake_mine(cfg, *, tag: str, resume: bool):
        captured["cfg"] = cfg
        captured["tag"] = tag
        captured["resume"] = resume
        return []

    monkeypatch.setattr(factor_lab.mining, "mine", fake_mine)

    parser = factor_lab.build_parser()
    args = parser.parse_args([
        "mine",
        "--tag",
        "unit_clean_split",
        "--rounds",
        "1",
        "--py-per-loop",
        "0",
        "--llm-per-loop",
        "0",
        "--train",
        "2023-05-15:2025-10-01",
        "--oos",
        "2025-10-01:2025-12-01",
        "--train3",
        "2023-05-15:2025-09-01",
        "--val3",
        "2025-09-01:2025-12-01",
        "--real-test3",
        "2025-12-01:2026-04-01",
        "--val-windows",
        "2025-09-01:2025-10-15;2025-10-15:2025-12-01",
        "--no-resume",
    ])
    args.func(args)

    cfg = captured["cfg"]
    assert captured["tag"] == "unit_clean_split"
    assert captured["resume"] is False
    assert cfg.train == ("2023-05-15", "2025-10-01")
    assert cfg.oos == ("2025-10-01", "2025-12-01")
    assert cfg.train3 == ("2023-05-15", "2025-09-01")
    assert cfg.val3 == ("2025-09-01", "2025-12-01")
    assert cfg.real_test3 == ("2025-12-01", "2026-04-01")
    assert cfg.val_windows == (
        ("2025-09-01", "2025-10-15"),
        ("2025-10-15", "2025-12-01"),
    )


def test_mine_cli_defaults_val_windows_to_custom_val3(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import factor_lab

    captured: dict[str, object] = {}

    def fake_mine(cfg, *, tag: str, resume: bool):
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(factor_lab.mining, "mine", fake_mine)

    parser = factor_lab.build_parser()
    args = parser.parse_args([
        "mine",
        "--rounds",
        "1",
        "--py-per-loop",
        "0",
        "--llm-per-loop",
        "0",
        "--val3",
        "2025-09-01:2025-12-01",
    ])
    args.func(args)

    cfg = captured["cfg"]
    assert cfg.val3 == ("2025-09-01", "2025-12-01")
    assert cfg.val_windows == (("2025-09-01", "2025-12-01"),)


def test_mine_cli_rejects_reversed_split_window() -> None:
    from scripts import factor_lab

    parser = factor_lab.build_parser()
    args = parser.parse_args(["mine", "--train", "2025-12-01:2025-10-01"])
    with pytest.raises(SystemExit, match="--train start must be before end"):
        args.func(args)

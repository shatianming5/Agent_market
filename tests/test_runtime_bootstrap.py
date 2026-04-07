from __future__ import annotations

from pathlib import Path


def test_ensure_repo_user_data_data_link_uses_release_sibling(tmp_path: Path) -> None:
    from agent_market.runtime_bootstrap import ensure_repo_user_data_data_link

    shared_repo = tmp_path / "Agent_market"
    release_repo = tmp_path / "Agent_market_release_123"
    shared_data = shared_repo / "user_data" / "data" / "gate"
    shared_data.mkdir(parents=True, exist_ok=True)
    (shared_data / "DOGE_USDT-1h.feather").write_text("stub", encoding="utf-8")
    (release_repo / "user_data").mkdir(parents=True, exist_ok=True)

    result = ensure_repo_user_data_data_link(repo_root=release_repo)

    assert result["status"] == "linked"
    linked = release_repo / "user_data" / "data"
    assert linked.is_symlink()
    assert linked.resolve() == (shared_repo / "user_data" / "data").resolve()


def test_ensure_repo_user_data_data_link_keeps_existing_local_data(tmp_path: Path) -> None:
    from agent_market.runtime_bootstrap import ensure_repo_user_data_data_link

    repo_root = tmp_path / "Agent_market"
    local_data = repo_root / "user_data" / "data" / "gate"
    local_data.mkdir(parents=True, exist_ok=True)
    (local_data / "ADA_USDT-1h.feather").write_text("stub", encoding="utf-8")

    result = ensure_repo_user_data_data_link(repo_root=repo_root)

    assert result["status"] == "present"
    assert not (repo_root / "user_data" / "data").is_symlink()

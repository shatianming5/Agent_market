from agent_market.freqai.runtime_threads import resolve_num_threads


def test_resolve_num_threads_prefers_explicit_config(monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_MODEL_THREADS", "8")
    assert resolve_num_threads({"num_threads": 3}) == 3
    assert resolve_num_threads({"nthread": 5}) == 5


def test_resolve_num_threads_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("AGENT_MARKET_MODEL_THREADS", "12")
    assert resolve_num_threads({}) == 12


def test_resolve_num_threads_uses_secondary_env(monkeypatch):
    monkeypatch.delenv("AGENT_MARKET_MODEL_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert resolve_num_threads({}) == 6


def test_resolve_num_threads_ignores_bad_values(monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_MODEL_THREADS", "bad")
    monkeypatch.setenv("OMP_NUM_THREADS", "-4")
    assert resolve_num_threads({}) == 1

from __future__ import annotations

from agent_market.microstructure.capture.kucoin import KuCoinLevel2SeqGapTracker


def test_kucoin_level2_seq_gap_tracker_no_gap_contiguous() -> None:
    t = KuCoinLevel2SeqGapTracker()
    topic = "/market/level2:BTC-USDT"
    t.observe(topic=topic, data={"symbol": "BTC-USDT", "sequenceStart": 100, "sequenceEnd": 101})
    t.observe(topic=topic, data={"symbol": "BTC-USDT", "sequenceStart": 102, "sequenceEnd": 103})

    meta = t.meta()
    assert meta["count"] == 0
    assert meta["examples"] == []
    assert meta["per_symbol"]["BTC-USDT"]["updates"] == 2
    assert meta["per_symbol"]["BTC-USDT"]["gaps"] == 0
    assert meta["per_symbol"]["BTC-USDT"]["last_seq_end"] == 103


def test_kucoin_level2_seq_gap_tracker_detects_gap_and_example() -> None:
    t = KuCoinLevel2SeqGapTracker(example_limit=2)
    topic = "/market/level2:BTC-USDT"
    t.observe(topic=topic, data={"symbol": "BTC-USDT", "sequenceStart": 100, "sequenceEnd": 101})
    t.observe(topic=topic, data={"symbol": "BTC-USDT", "sequenceStart": 105, "sequenceEnd": 106})

    meta = t.meta()
    assert meta["count"] == 1
    assert meta["per_symbol"]["BTC-USDT"]["gaps"] == 1
    ex = meta["examples"][0]
    assert ex["symbol"] == "BTC-USDT"
    assert ex["expected_start"] == 102
    assert ex["got_start"] == 105
    assert ex["got_end"] == 106


def test_kucoin_level2_seq_gap_tracker_tracks_symbols_independently() -> None:
    t = KuCoinLevel2SeqGapTracker()
    t.observe(topic="/market/level2:BTC-USDT", data={"symbol": "BTC-USDT", "sequenceStart": 10, "sequenceEnd": 10})
    t.observe(topic="/market/level2:ETH-USDT", data={"symbol": "ETH-USDT", "sequenceStart": 20, "sequenceEnd": 20})
    t.observe(topic="/market/level2:BTC-USDT", data={"symbol": "BTC-USDT", "sequenceStart": 12, "sequenceEnd": 12})

    meta = t.meta()
    assert meta["count"] == 1
    assert meta["per_symbol"]["BTC-USDT"]["gaps"] == 1
    assert meta["per_symbol"]["ETH-USDT"]["gaps"] == 0


def test_kucoin_level2_seq_gap_tracker_respects_example_limit() -> None:
    t = KuCoinLevel2SeqGapTracker(example_limit=1)
    topic = "/market/level2:BTC-USDT"
    t.observe(topic=topic, data={"symbol": "BTC-USDT", "sequenceStart": 1, "sequenceEnd": 1})
    t.observe(topic=topic, data={"symbol": "BTC-USDT", "sequenceStart": 3, "sequenceEnd": 3})
    t.observe(topic=topic, data={"symbol": "BTC-USDT", "sequenceStart": 5, "sequenceEnd": 5})

    meta = t.meta()
    assert meta["count"] == 2
    assert len(meta["examples"]) == 1


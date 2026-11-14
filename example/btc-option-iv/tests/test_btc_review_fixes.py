from __future__ import annotations

import asyncio
from dataclasses import replace
import pytest

from btc_option_iv import live
from btc_option_iv.kalman import FilterManager
from btc_option_iv.market import (
    FORWARD_SOURCE_MARK,
    MISSING_INDEX,
    ForwardData,
    Instrument,
    IndexQuote,
    OptionQuote,
    Rejection,
    normalize_deribit_quote,
)
from test_kalman import EXPIRY_MS, NOW_MS, quote


def test_partial_ticker_references_use_reference_ratio_discount():
    instrument = Instrument("BTC-27DEC24-100000-C", "call", 100_000.0, EXPIRY_MS, None)
    event = OptionQuote(
        instrument.instrument_name, NOW_MS, 0.04, 0.05, 1.0, 1.0,
        underlying_price=100_000.0, interest_rate=0.7,
    )
    references = ForwardData(101_000.0, NOW_MS, 99_000.0, NOW_MS, FORWARD_SOURCE_MARK)

    normalized = normalize_deribit_quote(event, instrument, references, as_of_ms=NOW_MS)

    assert normalized.discount == pytest.approx(99_000.0 / 100_000.0)
    assert "index / forward" in normalized.conversion


def test_filter_accepts_small_future_skew_and_ignores_zero_iv_for_bootstrap():
    manager = FilterManager()
    zero, instrument = quote(0.0, iv=0.0, timestamp=NOW_MS + 500, name="zero")
    manager.ingest(zero, instrument, now_ms=NOW_MS, now=0.0)
    assert manager.snapshots(now=0.0)[EXPIRY_MS].observations == ()

    positive_left, instrument = quote(-0.6, timestamp=NOW_MS + 500, name="positive-left")
    manager.ingest(positive_left, instrument, now_ms=NOW_MS, now=0.001)
    positive_right, instrument = quote(0.6, timestamp=NOW_MS + 500, name="positive-right")
    manager.ingest(positive_right, instrument, now_ms=NOW_MS, now=0.002)
    view = manager.snapshots(now=0.002)[EXPIRY_MS]
    assert view.status == "tracking"
    assert {observation.instrument for observation in view.observations} == {
        "positive-left", "positive-right"
    }


def test_discard_removes_stale_observation_from_snapshot():
    manager = FilterManager()
    row, instrument = quote(0.0, name="lost")
    manager.ingest(row, instrument, now_ms=NOW_MS, now=0.0)
    assert manager.snapshots(now=0.0)[EXPIRY_MS].observations

    manager.discard("lost", EXPIRY_MS, now_ms=NOW_MS, now=0.1)

    assert manager.snapshots(now=0.1)[EXPIRY_MS].observations == ()


def test_live_retries_after_malformed_rest_response(monkeypatch):
    async def scenario():
        controller = live.LiveFeedController(lambda snapshot: None)
        attempts = 0
        original_sleep = asyncio.sleep

        def fetch():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("Deribit response has no result")
            return [], []

        async def subscribe(*args, **kwargs):
            if False:
                yield None
            raise asyncio.CancelledError

        async def immediate_sleep(_seconds):
            await original_sleep(0)

        monkeypatch.setattr(live, "fetch_active_btc_options", fetch)
        monkeypatch.setattr(live, "subscribe_btc_option_market_data", subscribe)
        monkeypatch.setattr(live.asyncio, "sleep", immediate_sleep)
        with pytest.raises(asyncio.CancelledError):
            await controller._run()
        assert attempts == 2

    asyncio.run(scenario())


def test_live_missing_reference_clears_published_row_but_keeps_latest_quote(monkeypatch):
    async def scenario():
        controller = live.LiveFeedController(lambda snapshot: None)
        instrument = Instrument("live", "call", 100_000.0, EXPIRY_MS, None)
        monkeypatch.setattr(live, "fetch_active_btc_options", lambda: ([instrument], []))
        monkeypatch.setattr(live.time, "time", lambda: NOW_MS / 1000)
        first = OptionQuote("live", NOW_MS, 0.04, 0.05, 1.0, 1.0)
        newer = replace(first, timestamp_ms=NOW_MS + 1, bid_price=0.041)
        events = [first, newer, IndexQuote(NOW_MS + 2, 100_000.0)]

        async def event_stream(*args, **kwargs):
            for event in events:
                yield event
            raise asyncio.CancelledError

        valid_row, _ = quote(0.0, timestamp=NOW_MS, name="live")
        outcomes = iter(
            (
                valid_row,
                Rejection("live", MISSING_INDEX, "missing", NOW_MS + 1),
                valid_row,
            )
        )
        monkeypatch.setattr(live, "subscribe_btc_option_market_data", event_stream)
        monkeypatch.setattr(live, "evaluate_quote", lambda *args, **kwargs: next(outcomes))
        with pytest.raises(asyncio.CancelledError):
            await controller._run()
        assert "live" in controller.snapshot["rows"]
        assert controller.snapshot["latest_options"]["live"] == newer
        assert controller.filters.snapshots(now=asyncio.get_running_loop().time())[EXPIRY_MS].observations[0].instrument == "live"

    asyncio.run(scenario())


def test_wait_for_timeout_is_caught_as_asyncio_timeout(monkeypatch):
    from btc_option_iv import stream

    class Socket:
        async def recv(self):
            raise asyncio.TimeoutError

    class Connection:
        async def __aenter__(self):
            return Socket()

        async def __aexit__(self, *args):
            return False

    class WebSockets:
        def connect(self, *args, **kwargs):
            return Connection()

    async def scenario():
        monkeypatch.setitem(__import__("sys").modules, "websockets", WebSockets())
        async def noop(*args, **kwargs):
            return None
        monkeypatch.setattr(stream, "_rpc", noop)
        generator = stream.subscribe_btc_option_market_data([], refresh_seconds=30)
        assert [event async for event in generator] == []

    asyncio.run(scenario())

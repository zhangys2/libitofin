"""Regression tests for reviewed Deribit capture failures."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pyarrow.dataset as ds
import pytest

import deribit_capture.run as capture_run
from deribit_capture.run import Capture, parse_args
from deribit_capture.sink import ParquetSink


TS_MS = 1_791_995_000_000


def test_sink_parts_do_not_collide_between_instances_or_restarts(tmp_path):
    for price in (1.0, 2.0, 3.0):
        sink = ParquetSink(tmp_path, min_free_gb=0.0)
        sink.add(
            "index",
            {"ts_ms": TS_MS, "recv_ns": price, "index_name": "btc_usd", "price": price},
        )
        assert sink.flush_all() == 1

    rows = ds.dataset(tmp_path / "index", format="parquet", partitioning="hive").to_table().to_pylist()
    assert sorted(row["price"] for row in rows) == [1.0, 2.0, 3.0]
    parts = list((tmp_path / "index").rglob("part-*.parquet"))
    assert len(parts) == 3
    assert not list(tmp_path.rglob(".tmp-*"))


def _fake_clock(monkeypatch, *, start=0.0):
    now = [start]
    fake_time = SimpleNamespace(
        monotonic=lambda: now[0],
        time=lambda: 1_800_000_000.0,
        time_ns=lambda: 1_800_000_000_000_000_000,
        strftime=time.strftime,
        gmtime=time.gmtime,
    )
    monkeypatch.setattr(capture_run, "time", fake_time)
    return now


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        return None


class _FakeConnection:
    def __init__(self, websocket, *, on_enter=None):
        self.websocket = websocket
        self.on_enter = on_enter

    async def __aenter__(self):
        if self.on_enter is not None:
            self.on_enter()
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_capture_retries_rest_failure_before_connect(monkeypatch, tmp_path):
    import websockets

    capture = Capture(parse_args(["--out", str(tmp_path), "--min-free-gb", "0"]))
    calls = 0
    connects = 0

    def fetch_instruments(currency):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary REST failure")
        return []

    websocket = _FakeWebSocket()

    def connect(*args, **kwargs):
        nonlocal connects
        connects += 1
        return _FakeConnection(websocket, on_enter=capture.stop.set)

    class ImmediateStop:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        async def wait(self):
            return False

    capture.stop = ImmediateStop()
    monkeypatch.setattr(capture_run, "fetch_instruments", fetch_instruments)
    monkeypatch.setattr(capture_run, "_log", lambda message: None)
    monkeypatch.setattr(websockets, "connect", connect)

    assert asyncio.run(capture.run()) == 0
    assert calls == 2
    assert connects == 1


def _prepare_session(monkeypatch, tmp_path):
    import websockets

    capture = Capture(parse_args(["--out", str(tmp_path), "--min-free-gb", "0"]))
    websocket = _FakeWebSocket()

    async def universe():
        return []

    async def refresh_books():
        return None

    monkeypatch.setattr(capture, "_universe", universe)
    monkeypatch.setattr(capture, "_refresh_books", refresh_books)
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: _FakeConnection(websocket))
    return capture, websocket


def test_watchdog_expires_during_control_frame_traffic(monkeypatch, tmp_path):
    capture, _ = _prepare_session(monkeypatch, tmp_path)
    now = _fake_clock(monkeypatch)
    controls = iter(
        [
            json.dumps({"id": 1, "result": 30}),
            json.dumps({"method": "heartbeat", "params": {"type": "heartbeat"}}),
            json.dumps({"id": 2, "result": ["ticker.BTC-PERPETUAL.100ms"]}),
            json.dumps({"method": "heartbeat", "params": {"type": "heartbeat"}}),
        ]
    )

    async def next_control(awaitable, timeout):
        awaitable.close()
        now[0] += 10.0
        try:
            return next(controls)
        except StopIteration:
            pytest.fail("watchdog did not expire while only control frames arrived")

    monkeypatch.setattr(capture_run.asyncio, "wait_for", next_control)
    monkeypatch.setattr(capture_run, "_log", lambda message: None)

    with pytest.raises(TimeoutError, match="no market data"):
        asyncio.run(capture._session())


def test_default_book_snapshot_and_empty_trade_batch_refresh_watchdog_schema():
    book = {
        "method": "subscription",
        "params": {
            "channel": "book.BTC-PERPETUAL.none.10.100ms",
            "data": {
                "timestamp": TS_MS,
                "instrument_name": "BTC-PERPETUAL",
                "bids": [[67122.0, 22560.0]],
                "asks": [[67122.5, 26230.0]],
            },
        },
    }
    assert capture_run._is_market_data(book)
    assert capture_run._is_market_data(
        {"method": "subscription", "params": {"channel": "trades.BTC-PERPETUAL.100ms", "data": []}}
    )


def test_session_routes_default_book_snapshot_and_resets_watchdog(monkeypatch, tmp_path):
    capture, _ = _prepare_session(monkeypatch, tmp_path)
    now = _fake_clock(monkeypatch)
    frame = {
        "method": "subscription",
        "params": {
            "channel": "book.BTC-PERPETUAL.none.10.100ms",
            "data": {
                "timestamp": TS_MS,
                "instrument_name": "BTC-PERPETUAL",
                "bids": [[67122.0, 22560.0]],
                "asks": [[67122.5, 26230.0]],
            },
        },
    }
    receives = iter([json.dumps(frame), None, None, None, None])

    async def receive(awaitable, timeout):
        awaitable.close()
        now[0] += 10.0
        return next(receives)

    monkeypatch.setattr(capture_run.asyncio, "wait_for", receive)
    monkeypatch.setattr(capture_run, "_log", lambda message: None)
    with pytest.raises(TimeoutError, match="no market data"):
        asyncio.run(capture._session())
    assert capture.router.counts["perp_book"] == 1
    assert now[0] == 50.0


def test_malformed_book_and_control_frames_do_not_refresh_watchdog():
    triple_book = {
        "method": "subscription",
        "params": {
            "channel": "book.BTC-PERPETUAL.none.10.100ms",
            "data": {
                "timestamp": TS_MS,
                "instrument_name": "BTC-PERPETUAL",
                "bids": [["new", 67122.0, 22560.0]],
                "asks": [],
            },
        },
    }
    assert not capture_run._is_market_data(triple_book)
    assert not capture_run._is_market_data({"method": "heartbeat", "params": {"type": "heartbeat"}})
    assert capture_run._rejection_category(triple_book) == "book"
    assert capture_run._rejection_category({"method": "heartbeat"}) is None


def test_session_logs_rejected_market_data_without_resetting_watchdog(monkeypatch, tmp_path):
    capture, _ = _prepare_session(monkeypatch, tmp_path)
    now = _fake_clock(monkeypatch)
    bad = {
        "method": "subscription",
        "params": {
            "channel": "book.BTC-PERPETUAL.none.10.100ms",
            "data": {"timestamp": TS_MS, "instrument_name": "BTC-PERPETUAL", "bids": [["new", 1, 2]], "asks": []},
        },
    }
    logs = []

    async def receive(awaitable, timeout):
        awaitable.close()
        now[0] += 10.0
        return json.dumps(bad)

    monkeypatch.setattr(capture_run.asyncio, "wait_for", receive)
    monkeypatch.setattr(capture_run, "_log", logs.append)
    with pytest.raises(TimeoutError, match="no market data"):
        asyncio.run(capture._session())
    assert len(logs) == 4
    assert all("rejected market-data frames={'book': 1}" in entry for entry in logs)


def test_rejection_logging_is_aggregated_and_rate_limited(monkeypatch):
    messages = []
    monkeypatch.setattr(capture_run, "_log", messages.append)
    rejected = {"book": 3, "ticker": 2}
    assert capture_run._report_rejections(rejected, 9.0, 0.0) == 0.0
    assert not messages
    assert capture_run._report_rejections(rejected, 10.0, 0.0) == 10.0
    assert messages == ["rejected market-data frames={'book': 3, 'ticker': 2}"]
    assert rejected == {}


def test_session_catches_distinct_asyncio_timeout_exception(monkeypatch, tmp_path):
    capture, _ = _prepare_session(monkeypatch, tmp_path)
    now = _fake_clock(monkeypatch)

    class LegacyAsyncTimeout(Exception):
        pass

    async def timeout_receive(awaitable, timeout):
        awaitable.close()
        now[0] += 31.0
        raise LegacyAsyncTimeout

    monkeypatch.setattr(capture_run.asyncio, "TimeoutError", LegacyAsyncTimeout)
    monkeypatch.setattr(capture_run.asyncio, "wait_for", timeout_receive)

    with pytest.raises(TimeoutError, match="no market data"):
        asyncio.run(capture._session())


def test_rejected_subscription_retries_after_backoff_without_acknowledging(monkeypatch, tmp_path):
    capture = Capture(parse_args(["--out", str(tmp_path), "--min-free-gb", "0"]))
    now = _fake_clock(monkeypatch)
    monkeypatch.setattr(capture_run, "_log", lambda message: None)
    websocket = _FakeWebSocket()
    channel = "ticker.BTC-PERPETUAL.100ms"

    async def exercise():
        await capture._subscribe(websocket, [channel])
        first_id = websocket.sent[-1]["id"]
        assert capture._ack_subscribe({"id": first_id, "error": {"code": 10000}})
        assert channel not in capture._subscribed

        await capture._subscribe(websocket, [channel])
        assert len(websocket.sent) == 1

        now[0] += capture_run._SUBSCRIBE_RETRY_S
        await capture._subscribe(websocket, [channel])
        second_id = websocket.sent[-1]["id"]
        assert len(websocket.sent) == 2
        assert capture._ack_subscribe({"id": second_id, "result": [channel]})
        assert channel in capture._subscribed

        await capture._subscribe(websocket, [channel])
        assert len(websocket.sent) == 2

    asyncio.run(exercise())

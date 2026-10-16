"""Offline tests for the Deribit tick capture (no network, no itofin)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.dataset as ds
import pytest

from deribit_capture.report import build_report, gaps
from deribit_capture.router import Router, is_option
from deribit_capture.run import channels_for
from deribit_capture.sink import DiskLow, ParquetSink

OPT = "BTC-16OCT26-87000-C"
T0 = 1_791_995_000_000  # 2026-10-14 UTC, mid-day


def ticker(ts, bid=0.0001, bid_amt=41.9, ask=0.0002, ask_amt=30.0, mark=0.0002, name=OPT):
    return {
        "method": "subscription",
        "params": {
            "channel": f"ticker.{name}.100ms",
            "data": {
                "timestamp": ts,
                "instrument_name": name,
                "state": "open",
                "best_bid_price": bid,
                "best_bid_amount": bid_amt,
                "best_ask_price": ask,
                "best_ask_amount": ask_amt,
                "mark_price": mark,
                "mark_iv": 42.8,
                "bid_iv": 40.0,
                "ask_iv": 44.0,
                "index_price": 81745.4,
                "underlying_price": 81778.6,
                "underlying_index": "BTC-16OCT26",
                "open_interest": 824.9,
                "last_price": 0.0003,
                "greeks": {"delta": 0.016, "gamma": 2e-05, "vega": 2.2, "theta": -13.3, "rho": 0.05},
                "stats": {"volume": 143.2, "volume_usd": 12020.0},
            },
        },
    }


def read(root, table):
    return ds.dataset(root / table, format="parquet", partitioning="hive").to_table().to_pylist()


@pytest.fixture
def sink(tmp_path):
    return ParquetSink(tmp_path, flush_seconds=3600, min_free_gb=0.0)


def test_option_name_detection():
    assert is_option(OPT)
    assert not is_option("BTC-PERPETUAL")
    assert not is_option("BTC-30OCT26")


def test_quotes_written_only_when_top_of_book_changes(tmp_path, sink):
    router = Router(sink, state_interval_ms=10_000)
    router.handle(ticker(T0), 1)
    router.handle(ticker(T0 + 100, mark=0.00021), 2)  # mark moved, book did not
    router.handle(ticker(T0 + 200, bid_amt=50.0), 3)  # bid size changed
    router.handle(ticker(T0 + 300, ask=0.0003), 4)  # ask price changed
    sink.flush_all()
    rows = read(tmp_path, "quotes")
    assert [r["ts_ms"] for r in sorted(rows, key=lambda r: r["ts_ms"])] == [T0, T0 + 200, T0 + 300]
    assert rows[0]["instrument"] == OPT
    assert rows[0]["bid_iv"] == 40.0


def test_option_state_is_sampled_per_instrument(tmp_path, sink):
    router = Router(sink, state_interval_ms=10_000)
    router.handle(ticker(T0), 1)
    router.handle(ticker(T0 + 9_999), 2)
    router.handle(ticker(T0 + 10_000), 3)
    router.handle(ticker(T0, name="BTC-16OCT26-88000-C"), 4)
    sink.flush_all()
    rows = read(tmp_path, "option_state")
    assert len(rows) == 3
    first = next(r for r in rows if r["instrument"] == OPT and r["ts_ms"] == T0)
    assert first["delta"] == 0.016 and first["open_interest"] == 824.9
    assert first["underlying_index"] == "BTC-16OCT26"


def test_perpetual_ticker_goes_to_ticker_other(tmp_path, sink):
    router = Router(sink)
    msg = {
        "method": "subscription",
        "params": {
            "channel": "ticker.BTC-PERPETUAL.100ms",
            "data": {
                "timestamp": T0,
                "instrument_name": "BTC-PERPETUAL",
                "best_bid_price": 81776.0,
                "best_bid_amount": 100.0,
                "best_ask_price": 81776.5,
                "best_ask_amount": 90.0,
                "mark_price": 81775.3,
                "index_price": 81745.9,
                "funding_8h": 0.0001,
                "current_funding": 0.0,
                "state": "open",
            },
        },
    }
    router.handle(msg, 1)
    sink.flush_all()
    rows = read(tmp_path, "ticker_other")
    assert rows[0]["instrument"] == "BTC-PERPETUAL" and rows[0]["funding_8h"] == 0.0001
    assert not (tmp_path / "quotes").exists()


def test_trades_parse_dedupe_and_keep_unknown_fields(tmp_path, sink):
    router = Router(sink)
    trade = {
        "timestamp": T0,
        "price": 0.0003,
        "direction": "buy",
        "index_price": 81745.0,
        "instrument_name": OPT,
        "trade_seq": 10,
        "amount": 2.5,
        "mark_price": 0.0002,
        "iv": 41.0,
        "tick_direction": 1,
        "trade_id": 123,
        "combo_id": "XYZ",
    }
    msg = {"method": "subscription", "params": {"channel": "trades.option.BTC.100ms", "data": [trade, dict(trade)]}}
    router.handle(msg, 1)
    router.handle(msg, 2)  # replayed batch must not duplicate
    sink.flush_all()
    rows = read(tmp_path, "trades")
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "123" and rows[0]["direction"] == "buy" and rows[0]["iv"] == 41.0
    assert json.loads(rows[0]["extra_json"]) == {"combo_id": "XYZ"}


def test_book_snapshot_is_flattened_and_delta_frames_are_rejected(tmp_path, sink):
    router = Router(sink)
    snap = {
        "method": "subscription",
        "params": {
            "channel": "book.BTC-PERPETUAL.none.10.100ms",
            "data": {
                "timestamp": T0,
                "change_id": 7,
                "instrument_name": "BTC-PERPETUAL",
                "bids": [[81776.0, 10.0], [81775.0, 20.0]],
                "asks": [[81776.5, 5.0]],
            },
        },
    }
    delta = json.loads(json.dumps(snap))
    delta["params"]["data"]["bids"] = [["new", 81776.0, 10.0]]
    router.handle(snap, 1)
    router.handle(delta, 2)
    sink.flush_all()
    rows = read(tmp_path, "perp_book")
    assert len(rows) == 1
    assert rows[0]["bid_px"] == [81776.0, 81775.0] and rows[0]["ask_amt"] == [5.0]


def test_garbage_and_non_subscription_frames_are_ignored(sink):
    router = Router(sink)
    for message in ({}, {"id": 1, "result": []}, {"method": "subscription"}, {"method": "subscription", "params": {"channel": "ticker.x.100ms", "data": 5}}):
        router.handle(message, 1)
    assert sink.buffered_rows == 0


def test_sink_partitions_by_exchange_date_and_leaves_no_temp_files(tmp_path):
    sink = ParquetSink(tmp_path, min_free_gb=0.0)
    before_midnight = 1_791_935_999_000  # 2026-10-13 23:59:59 UTC
    after_midnight = before_midnight + 2_000
    sink.add("index", {"ts_ms": before_midnight, "recv_ns": 1, "index_name": "btc_usd", "price": 1.0})
    sink.add("index", {"ts_ms": after_midnight, "recv_ns": 2, "index_name": "btc_usd", "price": 2.0})
    assert sink.flush_all() == 2
    parts = sorted(p.name for p in (tmp_path / "index").iterdir())
    assert parts == ["date=2026-10-13", "date=2026-10-14"]
    assert not list(tmp_path.rglob(".tmp-*"))
    assert sink.buffered_rows == 0 and sink.rows_written == {"index": 2}


def test_sink_refuses_to_write_below_the_disk_guard_and_keeps_the_buffer(tmp_path):
    sink = ParquetSink(tmp_path, min_free_gb=10**9)
    sink.add("index", {"ts_ms": T0, "recv_ns": 1, "index_name": "btc_usd", "price": 1.0})
    with pytest.raises(DiskLow):
        sink.flush_all()
    assert sink.buffered_rows == 1


def test_sink_flushes_when_buffer_is_large(tmp_path):
    sink = ParquetSink(tmp_path, max_buffered_rows=2, flush_seconds=3600, min_free_gb=0.0)
    sink.add("index", {"ts_ms": T0, "recv_ns": 1, "index_name": "btc_usd", "price": 1.0})
    assert sink.maybe_flush() == 0
    sink.add("index", {"ts_ms": T0 + 1, "recv_ns": 2, "index_name": "btc_usd", "price": 2.0})
    assert sink.maybe_flush() == 2


def test_channels_include_core_streams_and_every_option():
    names = [OPT, "BTC-16OCT26-88000-P"]
    channels = channels_for(names, "BTC", book_depth=10)
    assert "trades.option.BTC.100ms" in channels
    assert "deribit_price_index.btc_usd" in channels
    assert "book.BTC-PERPETUAL.none.10.100ms" in channels
    assert f"ticker.{OPT}.100ms" in channels
    assert not any(c.startswith("book.") for c in channels_for(names, "BTC", book_depth=0))


def test_gap_detection():
    ts = np.array([0, 1_000, 2_000, 9_000, 10_000, 40_000], dtype=np.int64)
    assert gaps(ts, 5.0) == [(2_000, 9_000), (10_000, 40_000)]
    assert gaps(ts[:1], 5.0) == []


def test_report_summarises_coverage_and_drift(tmp_path):
    sink = ParquetSink(tmp_path, min_free_gb=0.0)
    for i, offset in enumerate([0, 1_000, 2_000, 30_000, 31_000]):
        ts = T0 + offset
        sink.add("index", {"ts_ms": ts, "recv_ns": (ts + 40) * 1_000_000, "index_name": "btc_usd", "price": 1.0})
    Router(sink).handle(ticker(T0), (T0 + 40) * 1_000_000)
    sink.flush_all()
    text = build_report(tmp_path, gap_seconds=5.0)
    assert "gaps > 5s: 1" in text
    assert "drift recv-minus-exchange ms: p50 40" in text
    assert "quotes: 1 distinct option instruments" in text

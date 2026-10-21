"""Offline tests for option order-book capture (no network)."""

from __future__ import annotations

import json

import pyarrow.dataset as ds
import pytest

from deribit_capture.router import Router
from deribit_capture.run import channels_for, parse_args, plan_book_channels, select_book_options
from deribit_capture.sink import ParquetSink

OPT = "BTC-16OCT26-87000-C"
T0 = 1_791_995_000_000
DAY_MS = 86_400_000


def read(root, table):
    return ds.dataset(root / table, format="parquet", partitioning="hive").to_table().to_pylist()


@pytest.fixture
def sink(tmp_path):
    return ParquetSink(tmp_path, flush_seconds=3600, min_free_gb=0.0)


def book(name, ts, bids, asks, change_id=1, depth=10):
    return {
        "method": "subscription",
        "params": {
            "channel": f"book.{name}.none.{depth}.100ms",
            "data": {
                "timestamp": ts,
                "change_id": change_id,
                "instrument_name": name,
                "bids": bids,
                "asks": asks,
            },
        },
    }


def test_option_book_goes_to_option_book_and_perp_book_is_unchanged(tmp_path, sink):
    router = Router(sink)
    router.handle(book(OPT, T0, [[0.0001, 41.9], [0.00005, 10.0]], [[0.0003, 135.9]]), 1)
    router.handle(book("BTC-PERPETUAL", T0, [[81776.0, 10.0]], [[81776.5, 5.0]]), 2)
    sink.flush_all()
    opt_rows = read(tmp_path, "option_book")
    perp_rows = read(tmp_path, "perp_book")
    assert len(opt_rows) == 1 and opt_rows[0]["instrument"] == OPT
    assert opt_rows[0]["bid_px"] == [0.0001, 0.00005] and opt_rows[0]["ask_amt"] == [135.9]
    assert len(perp_rows) == 1 and perp_rows[0]["instrument"] == "BTC-PERPETUAL"
    assert router.counts["option_book"] == 1 and router.counts["perp_book"] == 1


def test_identical_option_book_snapshots_are_not_stored_twice(tmp_path, sink):
    router = Router(sink)
    bids, asks = [[0.0001, 41.9]], [[0.0003, 135.9]]
    router.handle(book(OPT, T0, bids, asks, change_id=1), 1)
    router.handle(book(OPT, T0 + 100, bids, asks, change_id=2), 2)  # same content
    router.handle(book(OPT, T0 + 200, [[0.0001, 40.0]], asks, change_id=3), 3)  # size changed
    router.handle(book(OPT, T0 + 300, bids, asks, change_id=4), 4)  # back to the first content
    sink.flush_all()
    rows = sorted(read(tmp_path, "option_book"), key=lambda r: r["ts_ms"])
    assert [r["ts_ms"] for r in rows] == [T0, T0 + 200, T0 + 300]


def test_option_book_delta_frames_are_rejected(tmp_path, sink):
    router = Router(sink)
    frame = book(OPT, T0, [["new", 0.0001, 41.9]], [[0.0003, 135.9]])
    router.handle(frame, 1)
    sink.flush_all()
    assert not (tmp_path / "option_book").exists()


def test_empty_sides_are_kept_as_empty_lists(tmp_path, sink):
    router = Router(sink)
    router.handle(book(OPT, T0, [], [[0.0003, 135.9]]), 1)
    sink.flush_all()
    rows = read(tmp_path, "option_book")
    assert rows[0]["bid_px"] == [] and rows[0]["ask_px"] == [0.0003]


def test_channels_include_option_books_only_when_enabled():
    names = [OPT, "BTC-16OCT26-88000-P"]
    off = channels_for(names, "BTC", book_depth=10)
    assert not any(c.startswith("book.BTC-16OCT26") for c in off)
    on = channels_for(names, "BTC", book_depth=10, option_books=[OPT], option_book_depth=10)
    assert f"book.{OPT}.none.10.100ms" in on
    assert "book.BTC-16OCT26-88000-P.none.10.100ms" not in on
    assert "book.BTC-PERPETUAL.none.10.100ms" in on
    still_off = channels_for(names, "BTC", book_depth=10, option_books=[OPT], option_book_depth=0)
    assert f"book.{OPT}.none.10.100ms" not in still_off


def _instrument(name, strike, days, active=True):
    return {
        "instrument_name": name,
        "strike": strike,
        "expiration_timestamp": T0 + int(days * DAY_MS),
        "is_active": active,
    }


def _summary(name, forward, volume_usd, oi=0.0):
    return {"instrument_name": name, "underlying_price": forward, "volume_usd": volume_usd, "open_interest": oi}


def test_select_book_options_filters_and_ranks():
    instruments = [
        _instrument("A", 80_000, 2),
        _instrument("B", 90_000, 2),  # ln(90/80) = 0.118, inside the 0.15 band
        _instrument("C", 100_000, 2),  # ln(100/80) = 0.223, outside the band
        _instrument("D", 80_000, 60),  # too far out
        _instrument("E", 80_000, -1),  # already expired
        _instrument("F", 80_000, 2, active=False),
        _instrument("G", 80_000, 2),  # no summary row
        _instrument("H", 78_000, 5),  # same activity as B but closer to the money
    ]
    summaries = [
        _summary("A", 80_000, 5e6),
        _summary("B", 80_000, 1e6),
        _summary("C", 80_000, 9e6),
        _summary("D", 80_000, 9e6),
        _summary("E", 80_000, 9e6),
        _summary("F", 80_000, 9e6),
        _summary("H", 80_000, 1e6),
    ]
    picked = select_book_options(instruments, summaries, T0, max_days=35, band=0.15, top_n=10)
    assert picked == ["A", "H", "B"]
    assert select_book_options(instruments, summaries, T0, max_days=35, band=0.15, top_n=2) == ["A", "H"]


def test_select_book_options_handles_missing_or_bad_summary_data():
    instruments = [_instrument("A", 80_000, 2)]
    assert select_book_options(instruments, [], T0, max_days=35, band=0.15, top_n=5) == []
    bad = [{"instrument_name": "A", "underlying_price": None, "volume_usd": None}]
    assert select_book_options(instruments, bad, T0, max_days=35, band=0.15, top_n=5) == []


def test_plan_book_channels_subscribes_new_and_drops_stale_option_books_only():
    subscribed = {
        "ticker.X.100ms",
        "book.BTC-PERPETUAL.none.10.100ms",
        "book.BTC-16OCT26-80000-C.none.10.100ms",
        "book.BTC-16OCT26-81000-C.none.10.100ms",
    }
    to_sub, to_unsub = plan_book_channels(
        ["BTC-16OCT26-81000-C", "BTC-16OCT26-82000-C"], subscribed, depth=10
    )
    assert to_sub == ["book.BTC-16OCT26-82000-C.none.10.100ms"]
    assert to_unsub == ["book.BTC-16OCT26-80000-C.none.10.100ms"]


def test_option_book_is_off_by_default_and_flags_parse(tmp_path):
    args = parse_args(["--out", str(tmp_path)])
    assert args.option_book_depth == 0
    assert (args.option_book_max, args.option_book_days, args.option_book_band) == (200, 35.0, 0.15)
    args = parse_args(["--out", str(tmp_path), "--option-book-depth", "10", "--option-book-max", "50"])
    assert args.option_book_depth == 10 and args.option_book_max == 50

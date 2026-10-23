"""Unit tests for Hive-partitioned Level 1 Parquet recorder and dataset reader."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import pytest

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from spx_option_iv.recorder import (
    L1_RECORD_SCHEMA,
    ParquetL1Recorder,
    read_l1_dataset,
)
import spx_option_iv.feed as feed
from spx_option_iv.feed import (
    SpxFeedController,
    compute_0dte_expiry_ms,
)


EXPIRY_STR = "20261006"
CLOSE_MS = compute_0dte_expiry_ms(EXPIRY_STR)


def test_parquet_recorder_creates_hive_partitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        recorder = ParquetL1Recorder(output_dir=out_dir, flush_interval_sec=60.0)

        # Record SPX quotes
        recorder.record_quote(
            symbol="SPX",
            expiry="20261006",
            strike=7780.0,
            right="C",
            bid=12.50,
            ask=12.80,
            bid_size=10,
            ask_size=15,
            underlying_spot=7782.0,
            timestamp_ns=1_791_300_000_000_000_000,
        )
        recorder.record_quote(
            symbol="SPX",
            expiry="20261006",
            strike=7785.0,
            right="P",
            bid=10.20,
            ask=10.40,
            bid_size=20,
            ask_size=25,
            underlying_spot=7782.0,
            timestamp_ns=1_791_300_001_000_000_000,
        )

        # Record SPY quote
        recorder.record_quote(
            symbol="SPY",
            expiry="20261006",
            strike=775.0,
            right="C",
            bid=1.20,
            ask=1.25,
            bid_size=100,
            ask_size=150,
            underlying_spot=775.1,
            timestamp_ns=1_791_300_002_000_000_000,
        )

        # Flush to disk
        recorder.flush()
        recorder.close()

        # Verify directory structure
        spx_part_dir = out_dir / "symbol=SPX" / "expiry=20261006"
        assert spx_part_dir.exists()
        spx_date_dirs = list(spx_part_dir.glob("date=*"))
        assert len(spx_date_dirs) == 1
        spx_files = list(spx_date_dirs[0].glob("*.parquet"))
        assert len(spx_files) >= 1

        spy_part_dir = out_dir / "symbol=SPY" / "expiry=20261006"
        assert spy_part_dir.exists()
        spy_date_dirs = list(spy_part_dir.glob("date=*"))
        assert len(spy_date_dirs) == 1
        spy_files = list(spy_date_dirs[0].glob("*.parquet"))
        assert len(spy_files) >= 1

        # Read back with Hive dataset
        full_table = read_l1_dataset(out_dir)
        assert full_table.num_rows == 3

        # Verify partition pruning for SPY only
        spy_table = read_l1_dataset(out_dir, symbol="SPY")
        assert spy_table.num_rows == 1
        spy_dict = spy_table.to_pydict()
        assert spy_dict["strike"] == [775.0]
        assert spy_dict["symbol"] == ["SPY"]
        assert spy_dict["bid"] == [1.20]
        assert spy_dict["ask"] == [1.25]
        assert spy_dict["bid_size"] == [100]
        assert spy_dict["ask_size"] == [150]

        # Verify partition pruning for SPX only
        spx_table = read_l1_dataset(out_dir, symbol="SPX")
        assert spx_table.num_rows == 2
        spx_dict = spx_table.to_pydict()
        assert set(spx_dict["strike"]) == {7780.0, 7785.0}


def test_parquet_recorder_skips_invalid_quotes():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = ParquetL1Recorder(output_dir=tmpdir)

        # Inverted quote (ask < bid)
        recorder.record_quote(
            symbol="SPX", expiry="20261006", strike=7780.0, right="C",
            bid=15.0, ask=10.0, underlying_spot=7780.0
        )
        # None / NaN prices
        recorder.record_quote(
            symbol="SPX", expiry="20261006", strike=7780.0, right="C",
            bid=None, ask=10.0, underlying_spot=7780.0
        )
        recorder.record_quote(
            symbol="SPX", expiry="20261006", strike=7780.0, right="C",
            bid=10.0, ask=float("nan"), underlying_spot=7780.0
        )
        # Non-positive prices
        recorder.record_quote(
            symbol="SPX", expiry="20261006", strike=7780.0, right="C",
            bid=0.0, ask=10.0, underlying_spot=7780.0
        )

        assert recorder.buffered_records_count == 0
        recorder.flush()
        recorder.close()

        # No directories should have been created
        assert list(Path(tmpdir).glob("symbol=*")) == []


def test_parquet_recorder_auto_rotates_on_row_limit(monkeypatch):
    # Freeze the clock so both parts would share one millisecond name.
    monkeypatch.setattr("spx_option_iv.recorder.time.time", lambda: 1_700_000_000.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        # Low max_rows_per_file to trigger rotation
        recorder = ParquetL1Recorder(
            output_dir=out_dir,
            batch_size_limit=10,
            max_rows_per_file=5,
        )

        for i in range(12):
            recorder.record_quote(
                symbol="SPX",
                expiry="20261006",
                strike=7700.0 + i,
                right="C",
                bid=10.0,
                ask=11.0,
                underlying_spot=7700.0,
                timestamp_ns=1_000_000_000 + i,
            )
            # Flush after each batch
            if (i + 1) % 4 == 0:
                recorder.flush()

        recorder.close()

        # Check that multiple parquet files were rotated
        part_dir = next((out_dir / "symbol=SPX" / "expiry=20261006").glob("date=*"))
        files = list(part_dir.glob("*.parquet"))
        assert len(files) >= 2

        # Verify all rows are readable across part files
        table = read_l1_dataset(out_dir, symbol="SPX")
        assert table.num_rows == 12


def test_feed_controller_records_quotes_to_parquet(monkeypatch):
    """End-to-end integration: SpxFeedController streaming quotes into ParquetL1Recorder."""
    now_ms = CLOSE_MS - 3_600_000

    class _FakeTicker:
        def __init__(self, market_price=7740.0, bid=15.0, ask=15.5, bid_size=5, ask_size=10):
            self._market_price = market_price
            self.bid, self.ask = bid, ask
            self.bidSize, self.askSize = bid_size, ask_size
        def marketPrice(self):
            return self._market_price

    class _FakeIB:
        def __init__(self):
            self.option_contracts = []
        async def connectAsync(self, *args, **kwargs): pass
        def reqMarketDataType(self, kind): pass
        async def qualifyContractsAsync(self, *contracts): pass
        async def reqSecDefOptParamsAsync(self, *args):
            return [
                SimpleNamespace(
                    tradingClass="SPXW",
                    exchange="SMART",
                    expirations=[EXPIRY_STR],
                    strikes=[7735.0, 7740.0, 7745.0],
                )
            ]
        def reqMktData(self, contract, *args, **kwargs):
            if hasattr(contract, "symbol") and contract.symbol == "SPX" and hasattr(contract, "strike"):
                self.option_contracts.append(contract)
                return _FakeTicker(market_price=7740.0, bid=15.0, ask=15.5, bid_size=8, ask_size=12)
            return _FakeTicker(market_price=7740.0)
        def isConnected(self): return False

    class _FakeIndex:
        conId = 1
        def __init__(self, *args): pass

    class _FakeStock:
        conId = 756733
        def __init__(self, *args): pass

    class _FakeOption:
        def __init__(self, symbol, expiry, strike, right, exchange, tradingClass=None):
            self.symbol = symbol
            self.strike, self.right = float(strike), right
            self.exchange, self.tradingClass = exchange, tradingClass

    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = ParquetL1Recorder(output_dir=tmpdir, flush_interval_sec=60.0)

        fake_ib = _FakeIB()
        monkeypatch.setattr(
            feed, "_load_ib", lambda: (lambda: fake_ib, _FakeIndex, _FakeOption, _FakeStock)
        )
        monkeypatch.setattr(feed.time, "time", lambda: now_ms / 1000.0)

        sleep_count = {"n": 0}
        real_sleep = asyncio.sleep
        async def mock_sleep(d):
            sleep_count["n"] += 1
            if sleep_count["n"] > 2:
                raise asyncio.CancelledError
            await real_sleep(0)

        monkeypatch.setattr(feed.asyncio, "sleep", mock_sleep)

        controller = SpxFeedController(lambda s: None, symbol="SPX", recorder=recorder)
        try:
            asyncio.run(controller._run())
        except asyncio.CancelledError:
            pass

        # Close recorder and read saved parquet dataset
        recorder.close()
        table = read_l1_dataset(tmpdir, symbol="SPX")
        assert table.num_rows > 0
        d = table.to_pydict()
        assert d["symbol"][0] == "SPX"
        assert d["expiry"][0] == EXPIRY_STR
        assert d["underlying_spot"][0] == 7740.0
        assert d["bid"][0] == 15.0
        assert d["ask"][0] == 15.5
        assert d["bid_size"][0] == 8
        assert d["ask_size"][0] == 12


def test_read_l1_dataset_empty_dir():
    """Verify read_l1_dataset returns empty table on empty output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        table = read_l1_dataset(tmpdir, symbol="SPX")
        assert table.num_rows == 0
        assert "symbol" in table.column_names
        assert "strike" in table.column_names

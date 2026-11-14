from __future__ import annotations

import asyncio

import pytest

import spx_option_iv.recorder as recorder_module
from spx_option_iv.recorder import ParquetL1Recorder, read_l1_dataset


def _record_quote(recorder: ParquetL1Recorder, strike: float) -> None:
    recorder.record_quote(
        symbol="SPX",
        expiry="20261006",
        strike=strike,
        right="C",
        bid=1.0,
        ask=1.1,
        underlying_spot=100.0,
        timestamp_ns=1_000_000_000,
    )


def test_flush_parts_are_readable_while_recording_and_after_stop(tmp_path, monkeypatch):
    monkeypatch.setattr("spx_option_iv.recorder.time.time", lambda: 1_700_000_000.0)
    recorder = ParquetL1Recorder(output_dir=tmp_path, batch_size_limit=100)

    _record_quote(recorder, 100.0)
    recorder.flush()

    first_read = read_l1_dataset(tmp_path, symbol="SPX")
    assert first_read.num_rows == 1
    assert first_read["strike"].to_pylist() == [100.0]

    _record_quote(recorder, 101.0)
    recorder.flush()
    assert sorted(read_l1_dataset(tmp_path, symbol="SPX")["strike"].to_pylist()) == [
        100.0,
        101.0,
    ]

    asyncio.run(recorder.stop())
    assert read_l1_dataset(tmp_path, symbol="SPX").num_rows == 2


def test_flush_propagates_finalization_error_and_retains_rows(tmp_path, monkeypatch):
    recorder = ParquetL1Recorder(output_dir=tmp_path, batch_size_limit=100)
    _record_quote(recorder, 100.0)
    parquet_writer = recorder_module.pq.ParquetWriter

    class CloseFails:
        def __init__(self, *args, **kwargs):
            self.writer = parquet_writer(*args, **kwargs)

        def write_table(self, table):
            self.writer.write_table(table)

        def close(self):
            self.writer.close()
            raise OSError("Parquet finalization failed")

    monkeypatch.setattr(recorder_module.pq, "ParquetWriter", CloseFails)
    with pytest.raises(OSError, match="Parquet finalization failed"):
        recorder.flush()

    assert recorder.buffered_records_count == 1
    assert list(tmp_path.rglob("*.parquet")) == []

    monkeypatch.setattr(recorder_module.pq, "ParquetWriter", parquet_writer)
    recorder.flush()
    assert recorder.buffered_records_count == 0
    assert recorder.total_records_written == 1
    assert read_l1_dataset(tmp_path, symbol="SPX")["strike"].to_pylist() == [100.0]
    recorder.close()

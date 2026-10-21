"""High-performance Level 1 market data recorder with Hive-partitioned Parquet storage.

Partitioning layout:
    {output_dir}/
        symbol={symbol}/
            expiry={expiry}/
                date={date}/
                    quotes_{start_time_ms}.parquet

Because symbol, expiry, and date are encoded in the Hive directory paths,
they cost 0 bytes inside each Parquet file and enable instant partition
pruning when querying with DuckDB, Polars, PyArrow, or Pandas.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import math
from pathlib import Path
import threading, time
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Compact Arrow schema for Level 1 option quotes
# 'right' is dictionary-encoded (1-byte integer mapping to 'C' or 'P')
L1_RECORD_SCHEMA = pa.schema(
    [
        ("timestamp_ns", pa.int64()),
        ("strike", pa.float64()),
        ("right", pa.dictionary(pa.int8(), pa.string())),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("bid_size", pa.int32()),
        ("ask_size", pa.int32()),
        ("underlying_spot", pa.float64()),
    ]
)


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return f if math.isfinite(f) and f > 0 else None
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int:
    try:
        if val is None:
            return 0
        f = float(val)
        return int(f) if math.isfinite(f) and f >= 0 else 0
    except (ValueError, TypeError):
        return 0


class ParquetL1Recorder:
    """Buffers live Level 1 option quotes and writes Hive-partitioned Parquet files.

    Features:
    - Micro-batched memory buffer to prevent small-file/small-rowgroup fragmentation.
    - Zero-overhead Hive directory partitioning (symbol=.../expiry=.../date=...).
    - Row-group flushing with automatic file rotation after a configurable row threshold.
    - Robust shutdown flushing ensuring zero quote loss.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/l1",
        flush_interval_sec: float = 10.0,
        batch_size_limit: int = 10_000,
        max_rows_per_file: int = 100_000,
        compression: str = "zstd",
        compression_level: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.flush_interval_sec = flush_interval_sec
        self.batch_size_limit = batch_size_limit
        self.max_rows_per_file = max_rows_per_file
        self.compression = compression
        self.compression_level = compression_level
        self._buffer: dict[tuple[str, str, str], list[dict]] = {}
        self._active_writers: dict[
            tuple[str, str, str], tuple[pq.ParquetWriter, int, Path]
        ] = {}
        self._running = False
        self._flush_task: asyncio.Task | None = None
        self._total_records_written = 0
        self._lock = threading.RLock()

    @property
    def total_records_written(self) -> int:
        with self._lock:
            return self._total_records_written

    @property
    def buffered_records_count(self) -> int:
        with self._lock:
            return sum(len(records) for records in self._buffer.values())

    def record_quote(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        bid: Any,
        ask: Any,
        bid_size: Any = 0,
        ask_size: Any = 0,
        underlying_spot: Any = 0.0,
        timestamp_ns: int | None = None,
    ):
        """Append an incoming Level 1 quote to the in-memory buffer."""
        clean_bid = _safe_float(bid)
        clean_ask = _safe_float(ask)
        if clean_bid is None or clean_ask is None or clean_ask < clean_bid:
            return  # Reject invalid or inverted quotes

        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        date_str = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).strftime("%Y%m%d")
        partition_key = (symbol.upper(), str(expiry), date_str)
        record = {
            "timestamp_ns": int(ts),
            "strike": float(strike),
            "right": right.upper()[:1],
            "bid": clean_bid,
            "ask": clean_ask,
            "bid_size": _safe_int(bid_size),
            "ask_size": _safe_int(ask_size),
            "underlying_spot": float(underlying_spot or 0.0),
        }

        with self._lock:
            if partition_key not in self._buffer:
                self._buffer[partition_key] = []
            self._buffer[partition_key].append(record)
            if self.buffered_records_count >= self.batch_size_limit:
                self.flush()

    def flush(self):
        """Flush all buffered records into active partition Parquet writers."""
        with self._lock:
            if not self._buffer:
                return
            pending = self._buffer
            self._buffer = {}
            remaining = dict(pending)
            try:
                for partition_key, records in list(remaining.items()):
                    if not records:
                        remaining.pop(partition_key)
                        continue
                    writer, row_count, _file_path = self._get_or_create_writer(partition_key)
                    table = pa.Table.from_pylist(records, schema=L1_RECORD_SCHEMA)
                    writer.write_table(table)
                    new_count = row_count + len(records)
                    self._total_records_written += len(records)
                    if new_count >= self.max_rows_per_file:
                        writer.close()
                        del self._active_writers[partition_key]
                    else:
                        self._active_writers[partition_key] = (
                            writer,
                            new_count,
                            self._active_writers[partition_key][2],
                        )
                    remaining.pop(partition_key)
            except Exception:
                for partition_key, records in remaining.items():
                    self._buffer.setdefault(partition_key, [])
                    self._buffer[partition_key] = list(records) + self._buffer[partition_key]
                raise

    def _get_or_create_writer(
        self, partition_key: tuple[str, str, str]
    ) -> tuple[pq.ParquetWriter, int, Path]:
        if partition_key in self._active_writers:
            return self._active_writers[partition_key]

        symbol, expiry, date_str = partition_key
        partition_dir = (
            self.output_dir
            / f"symbol={symbol}"
            / f"expiry={expiry}"
            / f"date={date_str}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        now_ms = int(time.time() * 1000)
        file_path = partition_dir / f"quotes_{now_ms}.parquet"

        writer = pq.ParquetWriter(
            file_path,
            schema=L1_RECORD_SCHEMA,
            compression=self.compression,
            compression_level=self.compression_level,
            use_dictionary=True,
        )
        entry = (writer, 0, file_path)
        self._active_writers[partition_key] = entry
        return entry

    async def record_quote_async(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        bid: Any,
        ask: Any,
        bid_size: Any = 0,
        ask_size: Any = 0,
        underlying_spot: Any = 0.0,
        timestamp_ns: int | None = None,
    ):
        """Record one quote without blocking the calling asyncio loop."""
        operation = asyncio.create_task(
            asyncio.to_thread(
                self.record_quote,
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                underlying_spot=underlying_spot,
                timestamp_ns=timestamp_ns,
            )
        )
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            await operation
            raise

    async def flush_async(self):
        """Flush buffered Parquet rows without blocking the calling asyncio loop."""
        await asyncio.to_thread(self.flush)

    def close(self):
        """Flush remaining buffer and close all active Parquet files."""
        self._running = False
        task = self._flush_task
        if task is not None:
            if not task.done():
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is task.get_loop():
                    task.cancel()
                else:
                    task.get_loop().call_soon_threadsafe(task.cancel)
            self._flush_task = None
        with self._lock:
            self.flush()
            for writer, _, _ in self._active_writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            self._active_writers.clear()

    async def _flush_loop(self):
        """Background loop to periodically flush micro-batches."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_sec)
                await self.flush_async()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def start(self):
        """Start background asynchronous flushing loop."""
        if not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        """Stop background flushing loop and cleanly close all writers."""
        self._running = False
        task = self._flush_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if self._flush_task is task:
                self._flush_task = None
        await asyncio.to_thread(self.close)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


PARTITION_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("expiry", pa.string()),
        ("date", pa.string()),
    ]
)

FULL_L1_SCHEMA = pa.schema([*PARTITION_SCHEMA, *L1_RECORD_SCHEMA])


def read_l1_dataset(
    output_dir: str | Path,
    symbol: str | None = None,
    expiry: str | None = None,
    date: str | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Read Level 1 data using PyArrow dataset with Hive partition pruning.

    Filters are evaluated against partition directory paths without scanning
    unmatched files.
    """
    path = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path {path} does not exist")

    partitioning = ds.partitioning(schema=PARTITION_SCHEMA, flavor="hive")
    dataset = ds.dataset(
        path,
        schema=FULL_L1_SCHEMA,
        partitioning=partitioning,
        format="parquet",
    )
    filter_expr = None

    if symbol is not None:
        expr = ds.field("symbol") == symbol.upper()
        filter_expr = expr if filter_expr is None else (filter_expr & expr)

    if expiry is not None:
        expr = ds.field("expiry") == str(expiry)
        filter_expr = expr if filter_expr is None else (filter_expr & expr)

    if date is not None:
        expr = ds.field("date") == str(date)
        filter_expr = expr if filter_expr is None else (filter_expr & expr)

    return dataset.to_table(filter=filter_expr, columns=columns)

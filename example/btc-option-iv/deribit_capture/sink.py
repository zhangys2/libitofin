"""Buffered, crash-safe Parquet sink partitioned by table and UTC date.

Layout::

    {root}/{table}/date=YYYY-MM-DD/part-{first_ts_ms}-{seq}.parquet

Each flush writes a complete file to a temporary name and renames it, so a
reader never sees a half-written file and a crash loses at most the unflushed
buffer (one flush interval).
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from deribit_capture.schemas import SCHEMAS


class DiskLow(RuntimeError):
    """Raised when free space under the output root falls below the guard."""


class ParquetSink:
    def __init__(
        self,
        root: str | Path,
        *,
        flush_seconds: float = 60.0,
        max_buffered_rows: int = 200_000,
        compression: str = "zstd",
        compression_level: int = 6,
        min_free_gb: float = 10.0,
    ) -> None:
        self.root = Path(root)
        self.flush_seconds = flush_seconds
        self.max_buffered_rows = max_buffered_rows
        self.compression = compression
        self.compression_level = compression_level
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self._cols: dict[tuple[str, str], dict[str, list]] = {}
        self._rows: dict[tuple[str, str], int] = {}
        self._first_ts: dict[tuple[str, str], int] = {}
        self._seq = 0
        self._last_flush = time.monotonic()
        self.rows_written: dict[str, int] = {}
        self.bytes_written = 0

    @property
    def buffered_rows(self) -> int:
        return sum(self._rows.values())

    def add(self, table: str, row: dict) -> None:
        """Buffer one row. ``row`` must carry ``ts_ms``; missing columns become null."""
        date = datetime.fromtimestamp(row["ts_ms"] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        key = (table, date)
        cols = self._cols.get(key)
        if cols is None:
            cols = {name: [] for name in SCHEMAS[table].names}
            self._cols[key] = cols
            self._rows[key] = 0
            self._first_ts[key] = row["ts_ms"]
        for name, bucket in cols.items():
            bucket.append(row.get(name))
        self._rows[key] += 1

    def maybe_flush(self) -> int:
        """Flush when the interval elapsed or the buffer is large. Returns rows written."""
        if (
            time.monotonic() - self._last_flush >= self.flush_seconds
            or self.buffered_rows >= self.max_buffered_rows
        ):
            return self.flush_all()
        return 0

    def flush_all(self) -> int:
        self._last_flush = time.monotonic()
        if not self._cols:
            return 0
        self._check_disk()
        written = 0
        for key in list(self._cols):
            written += self._flush_key(key)
        return written

    def _flush_key(self, key: tuple[str, str]) -> int:
        table_name, date = key
        cols = self._cols.get(key)
        count = self._rows.get(key, 0)
        first_ts = self._first_ts.get(key)
        if cols is None or count == 0 or first_ts is None:
            self._cols.pop(key, None)
            self._rows.pop(key, None)
            self._first_ts.pop(key, None)
            return 0
        table = pa.Table.from_pydict(cols, schema=SCHEMAS[table_name])
        directory = self.root / table_name / f"date={date}"
        directory.mkdir(parents=True, exist_ok=True)
        self._seq += 1
        final = directory / f"part-{first_ts}-{self._seq:06d}.parquet"
        temp = directory / f".tmp-{final.name}"
        pq.write_table(
            table,
            temp,
            compression=self.compression,
            compression_level=self.compression_level,
        )
        os.replace(temp, final)
        self._cols.pop(key, None)
        self._rows.pop(key, None)
        self._first_ts.pop(key, None)
        self.rows_written[table_name] = self.rows_written.get(table_name, 0) + count
        self.bytes_written += final.stat().st_size
        return count

    def _check_disk(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        if free < self.min_free_bytes:
            raise DiskLow(
                f"only {free / 1024**3:.1f} GiB free under {self.root}; "
                f"guard is {self.min_free_bytes / 1024**3:.1f} GiB"
            )

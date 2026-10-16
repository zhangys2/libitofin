"""Summarise a capture directory: size, coverage, gaps and clock drift.

Usage:
    python -m deribit_capture.report --root ~/vol-rv/data/deribit_ticks [--gap-seconds 5]

Coverage and gaps are measured on the ``index`` table, which ticks about once a
second regardless of option activity, so a gap there means the capture was down
or disconnected. Drift is local receive time minus the exchange timestamp.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


def _dataset(root: Path, table: str):
    path = root / table
    if not path.exists() or not any(path.rglob("*.parquet")):
        return None
    return ds.dataset(path, format="parquet", partitioning="hive")


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def table_sizes(root: Path) -> list[tuple[str, int, int, str, str]]:
    rows = []
    for name in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "instruments"):
        files = [f for f in (root / name).rglob("*.parquet") if not f.name.startswith(".tmp-")]
        if not files:
            continue
        size = sum(f.stat().st_size for f in files)
        dates = sorted({f.parent.name.removeprefix("date=") for f in files})
        dataset = _dataset(root, name)
        count = dataset.count_rows() if dataset else 0
        rows.append((name, count, size, dates[0], dates[-1]))
    return rows


def gaps(ts_ms: np.ndarray, threshold_s: float) -> list[tuple[int, int]]:
    """(start_ms, end_ms) of every pause longer than ``threshold_s``."""
    if ts_ms.size < 2:
        return []
    ordered = np.sort(ts_ms)
    delta = np.diff(ordered)
    idx = np.nonzero(delta > threshold_s * 1000.0)[0]
    return [(int(ordered[i]), int(ordered[i + 1])) for i in idx]


def build_report(root: Path, gap_seconds: float) -> str:
    out: list[str] = [f"capture root: {root}"]
    sizes = table_sizes(root)
    if not sizes:
        return "\n".join(out + ["no data"])
    out.append("")
    out.append(f"{'table':<14}{'rows':>14}{'MB':>10}  dates")
    for name, count, size, first, last in sizes:
        out.append(f"{name:<14}{count:>14,}{size / 1e6:>10.1f}  {first} .. {last}")

    index = _dataset(root, "index")
    if index is not None:
        table = index.to_table(columns=["ts_ms", "recv_ns"])
        ts = table.column("ts_ms").to_numpy()
        recv_ms = table.column("recv_ns").to_numpy() / 1e6
        span_s = (ts.max() - ts.min()) / 1000.0
        found = gaps(ts, gap_seconds)
        down_s = sum(b - a for a, b in found) / 1000.0
        out.append("")
        out.append(f"coverage (index ticks): {_iso(int(ts.min()))} .. {_iso(int(ts.max()))} UTC, span {span_s / 86400:.2f} days")
        out.append(f"gaps > {gap_seconds:g}s: {len(found)}, total {down_s / 60:.1f} min, uptime {100 * (1 - down_s / span_s):.2f}%" if span_s > 0 else "span too short")
        for a, b in sorted(found, key=lambda g: g[1] - g[0], reverse=True)[:10]:
            out.append(f"  {_iso(a)} -> {_iso(b)}  ({(b - a) / 1000:.0f}s)")
        drift = recv_ms - ts
        out.append(
            f"drift recv-minus-exchange ms: p50 {np.percentile(drift, 50):.0f}, "
            f"p99 {np.percentile(drift, 99):.0f}, min {drift.min():.0f}, max {drift.max():.0f}"
        )

    quotes = _dataset(root, "quotes")
    if quotes is not None:
        names = quotes.to_table(columns=["instrument"]).column("instrument").unique()
        out.append("")
        out.append(f"quotes: {len(names)} distinct option instruments")
    trades = _dataset(root, "trades")
    if trades is not None:
        t = trades.to_table(columns=["instrument"])
        instruments = np.array(t.column("instrument").to_pylist())
        is_opt = np.array([i.endswith(("-C", "-P")) for i in instruments])
        out.append(f"trades: {int(is_opt.sum()):,} option, {int((~is_opt).sum()):,} other")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--gap-seconds", type=float, default=5.0)
    args = p.parse_args(argv)
    print(build_report(Path(args.root).expanduser(), args.gap_seconds))


if __name__ == "__main__":
    main()

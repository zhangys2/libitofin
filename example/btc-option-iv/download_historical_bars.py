"""Historical 5-Minute Bar Downloader for Stocks & ETFs using IB Gateway.

Downloads 2 years of 5-minute OHLCV trade bars for a list of stocks (e.g. 20 liquid names)
from IB Gateway and saves them into high-performance, compressed Parquet files.

Features:
- Automatically stitches two 1-year chunks (IBKR max single duration for 5-min bars is 1 Y).
- Formats explicit timezone strings ('YYYYMMDD 16:00:00 US/Eastern') to avoid HMDS hangs.
- Pacing protection: 120s timeout and courteous inter-request sleep to avoid pacing violations.
- Saves per-symbol Parquet files with Zstandard (zstd) compression (~600-800 KB per stock for 2Y).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Sequence

# Initialize loop policy before importing eventkit/ib_async
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_async import Contract, IB, Stock, util  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# Default 20 liquid US stocks & ETFs
DEFAULT_STOCKS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "TSLA", "AMD", "JPM", "V", "UNH",
    "XOM", "JNJ", "PG", "HD", "COST",
]


async def fetch_2y_5m_bars_for_symbol(
    ib: IB,
    contract: Contract,
    bar_size: str = "5 mins",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    inter_chunk_pause: float = 3.0,
) -> pd.DataFrame:
    """Fetch 2 years of 5-minute bars by stitching two 1-year requests."""
    sym = contract.symbol
    print(f"[{sym}] Requesting Chunk 1: Recent 1 Year (up to now)...")

    bars_chunk1 = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr="1 Y",
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
        timeout=120,
    )
    if not bars_chunk1:
        print(f"[{sym}] [WARN] No bars returned for Chunk 1")
        return pd.DataFrame()

    c1_first = bars_chunk1[0].date
    c1_last = bars_chunk1[-1].date
    print(f"[{sym}] Chunk 1: {len(bars_chunk1):,} bars ({c1_first} to {c1_last})")

    # Format the earliest date of Chunk 1 as endDateTime with US/Eastern timezone
    # E.g. '20251007 16:00:00 US/Eastern'
    earliest_date_str = c1_first.strftime("%Y%m%d 16:00:00 US/Eastern")
    print(f"[{sym}] Requesting Chunk 2: Prior 1 Year (ending {earliest_date_str})...")

    await asyncio.sleep(inter_chunk_pause)

    bars_chunk2 = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime=earliest_date_str,
        durationStr="1 Y",
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
        timeout=120,
    )

    if bars_chunk2:
        c2_first = bars_chunk2[0].date
        c2_last = bars_chunk2[-1].date
        print(f"[{sym}] Chunk 2: {len(bars_chunk2):,} bars ({c2_first} to {c2_last})")
        df2 = util.df(bars_chunk2)
    else:
        print(f"[{sym}] [WARN] No bars returned for Chunk 2")
        df2 = pd.DataFrame()

    df1 = util.df(bars_chunk1)

    # Combine, deduplicate, and sort
    frames = [f for f in (df2, df1) if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["symbol"] = sym
    return df


async def main():
    parser = argparse.ArgumentParser(description="Download 2 years of 5-minute historical bars to Parquet")
    parser.add_argument("--host", default="127.0.0.1", help="IB Gateway/TWS host")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway port (4002 live, 4001 paper)")
    parser.add_argument("--client-id", type=int, default=70, help="API client ID")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_STOCKS,
        help="List of stock/ETF symbols (default: 20 liquid names)",
    )
    parser.add_argument("--output-dir", default="data/bars_5m", help="Directory to save Parquet files")
    parser.add_argument("--pause-between-stocks", type=float, default=2.0, help="Pause between stock downloads")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ib = IB()
    print(f"Connecting to IB Gateway at {args.host}:{args.port} (clientId={args.client_id})...")
    connected = False
    for cid in range(args.client_id, args.client_id + 15):
        try:
            await ib.connectAsync(args.host, args.port, clientId=cid, timeout=5, readonly=True)
            print(f"Connected to IB Gateway! (clientId={cid})")
            connected = True
            break
        except Exception:
            continue

    if not connected:
        print("ERROR: Could not connect to IB Gateway. Ensure gateway is running.")
        sys.exit(1)

    def on_error(req_id: int, error_code: int, error_string: str, contract: Contract):
        if error_code in (2104, 2106, 2158):
            return
        sym = getattr(contract, "symbol", "") if contract else ""
        print(f"[IBKR {error_code}] ({sym}): {error_string}")

    ib.errorEvent += on_error

    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    print(f"\nTargeting {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Output directory: {out_dir.resolve()}\n")

    start_total_time = time.time()
    success_count = 0

    for idx, sym in enumerate(symbols, start=1):
        print(f"\n{'='*70}\n[{idx}/{len(symbols)}] Processing {sym}...")
        try:
            contract = Stock(sym, exchange="SMART", currency="USD")
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                print(f"[{sym}] ERROR: Could not qualify contract with IBKR. Skipping.")
                continue

            t0 = time.time()
            df = await fetch_2y_5m_bars_for_symbol(ib, contract)

            if df.empty:
                print(f"[{sym}] ERROR: No data retrieved. Skipping save.")
                continue

            # Convert to PyArrow Table and save
            fpath = out_dir / f"{sym}.parquet"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, fpath, compression="zstd")

            elapsed = time.time() - t0
            size_kb = fpath.stat().st_size / 1024
            print(f"[{sym}] Saved {len(df):,} bars to {fpath} ({size_kb:.1f} KB) in {elapsed:.1f}s")
            success_count += 1

            if idx < len(symbols):
                await asyncio.sleep(args.pause_between_stocks)

        except Exception as exc:
            print(f"[{sym}] Failed with exception: {exc}")

    total_elapsed = time.time() - start_total_time
    print(f"\n{'='*70}\nFinished! Successfully saved {success_count}/{len(symbols)} symbols in {total_elapsed / 60:.1f} minutes.")
    ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

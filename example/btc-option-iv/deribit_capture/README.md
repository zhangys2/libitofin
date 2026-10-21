# deribit_capture

Raw tick capture from Deribit's **public** WebSocket (no API key). It exists to
build the dataset that quote, fill and hedge research needs; it does not depend
on `itofin`.

```bash
python -m deribit_capture --out ~/vol-rv/data/deribit_ticks     # run until SIGINT/SIGTERM
python -m deribit_capture.report --root ~/vol-rv/data/deribit_ticks
```

Dependencies: `pyarrow`, `numpy` (report only), `websockets`.

## Tables (Hive-partitioned Parquet, zstd: `{root}/{table}/date=YYYY-MM-DD/part-*.parquet`)

| Table | One row per | Notes |
|---|---|---|
| `quotes` | change in top-of-book price or size of an option | Deribit sends ~960 ticker frames/s but only ~125/s change the book; the rest are mark/Greek moves. |
| `option_state` | option, at most every `--state-interval` s (default 10) | mark, IVs, Greeks, open interest, volume, index, forward. |
| `trades` | trade | options and the perpetual; unknown fields kept in `extra_json`. |
| `ticker_other` | perpetual ticker frame | includes funding. |
| `perp_book` | perpetual book snapshot (top `--book-depth`, default 10) | list columns. |
| `option_book` | option book snapshot, written only when the content changes (top `--option-book-depth` levels) | list columns, same layout as `perp_book`. Off by default; see below. |
| `index` | index tick | ~1/s; used to measure uptime and clock drift. |
| `sessions` | connect / disconnect / error | gaps are discoverable afterwards. |
| `instruments/` | JSON snapshot of the listed options | saved whenever the listing changes. |

Prices are in coin (BTC) as Deribit quotes inverse options, IVs in percent.
`bid_px = 0` with `bid_amt = 0` means no quote on that side.

## Safety

- Writes are atomic (temp file, then rename); a crash loses at most one flush interval (`--flush-seconds`, default 60).
- The disk guard (`--min-free-gb`) stops the capture instead of filling the disk.
- Reconnects with backoff and restarts the subscription list; also notices when no market data arrives for 30 s. Heartbeats alone do not reset that timer.

## Known limits

- `recv_ns` is local receive time; the report's drift figure mixes network latency with clock offset.
- Only public data. Your own order and fill events are a separate stage.

## Option order books (optional)

`--option-book-depth 10` subscribes to the public `book.{option}.none.10.100ms` channel for a selected set of
options and writes the `option_book` table. Selection is refreshed every `--refresh-seconds` from Deribit's
instrument list and `get_book_summary_by_currency`: active options expiring within `--option-book-days`
(default 35) whose strike is within `--option-book-band` (default 0.15, log distance) of their forward, ranked
by 24h USD volume, then open interest, then closeness to the money, keeping the first `--option-book-max`
(default 200). Channels for options that drop out of the selection are unsubscribed. If the summary request
fails the previous selection is kept. Deribit sends no order IDs, so queue position behind displayed size is
still unknowable.

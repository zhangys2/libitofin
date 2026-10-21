"""Long-running Deribit public WebSocket capture. Public data only; no API key.

Usage:
    python -m deribit_capture --out ~/vol-rv/data/deribit_ticks
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import signal
import sys
import time
import urllib.request
from pathlib import Path

from deribit_capture.router import Router, is_option
from deribit_capture.sink import DiskLow, ParquetSink

WS_URL = "wss://www.deribit.com/ws/api/v2"
REST = "https://www.deribit.com/api/v2/public/get_instruments?currency={cur}&kind=option&expired=false"
REST_SUMMARY = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={cur}&kind=option"
_DAY_MS = 86_400_000
_CHUNK = 100
_SILENCE_S = 30.0


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", file=sys.stderr, flush=True)


def fetch_instruments(currency: str) -> list[dict]:
    request = urllib.request.Request(
        REST.format(cur=currency), headers={"User-Agent": "deribit-capture/0.1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    result = payload.get("result")
    if not isinstance(result, list):
        raise ValueError("get_instruments returned no result list")
    return result


def fetch_book_summaries(currency: str) -> list[dict]:
    request = urllib.request.Request(
        REST_SUMMARY.format(cur=currency), headers={"User-Agent": "deribit-capture/0.1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    result = payload.get("result")
    if not isinstance(result, list):
        raise ValueError("get_book_summary_by_currency returned no result list")
    return result


def _book_channel(name: str, depth: int) -> str:
    return f"book.{name}.none.{depth}.100ms"


def select_book_options(
    instruments: list[dict],
    summaries: list[dict],
    now_ms: int,
    *,
    max_days: float,
    band: float,
    top_n: int,
) -> list[str]:
    """Pick the liquid near-the-money options whose order book is worth recording.

    An option qualifies when it is active, expires within ``max_days``, and its strike is within
    ``band`` (log distance) of its own forward. Qualifying options are ranked by 24h USD volume, then
    open interest, then closeness to the money, and the first ``top_n`` are returned.
    """
    by_name = {s.get("instrument_name"): s for s in summaries if isinstance(s, dict)}
    ranked = []
    for row in instruments:
        name = row.get("instrument_name")
        strike = row.get("strike")
        expiry = row.get("expiration_timestamp")
        summary = by_name.get(name)
        if not row.get("is_active", True) or summary is None:
            continue
        forward = summary.get("underlying_price")
        if not isinstance(strike, (int, float)) or not isinstance(expiry, (int, float)):
            continue
        if isinstance(forward, bool) or not isinstance(forward, (int, float)) or forward <= 0:
            continue
        days = (expiry - now_ms) / _DAY_MS
        distance = abs(math.log(strike / forward)) if strike > 0 else math.inf
        if not 0.0 < days <= max_days or distance > band:
            continue
        volume = summary.get("volume_usd")
        interest = summary.get("open_interest")
        volume = float(volume) if isinstance(volume, (int, float)) and not isinstance(volume, bool) else 0.0
        interest = float(interest) if isinstance(interest, (int, float)) and not isinstance(interest, bool) else 0.0
        ranked.append((-volume, -interest, distance, name))
    ranked.sort()
    return [name for *_, name in ranked[:top_n]]


def stale_channels(subscribed: set[str], desired: list[str]) -> list[str]:
    """Subscribed channels that are no longer in the desired set, including option tickers."""
    keep = set(desired)
    return sorted(channel for channel in subscribed if channel not in keep)


def plan_book_channels(wanted: list[str], subscribed: set[str], *, depth: int) -> tuple[list[str], list[str]]:
    """Option-book channels to subscribe and to drop; perpetual and ticker channels are never dropped."""
    wanted_channels = [_book_channel(name, depth) for name in wanted]
    to_sub = [c for c in wanted_channels if c not in subscribed]
    keep = set(wanted_channels)
    to_unsub = sorted(
        c
        for c in subscribed
        if c.startswith("book.") and is_option(c.split(".")[1]) and c not in keep
    )
    return to_sub, to_unsub


def channels_for(
    option_names: list[str],
    currency: str,
    *,
    book_depth: int,
    option_books: list[str] = (),
    option_book_depth: int = 0,
) -> list[str]:
    perp = f"{currency}-PERPETUAL"
    fixed = [
        f"deribit_price_index.{currency.lower()}_usd",
        f"ticker.{perp}.100ms",
        f"trades.{perp}.100ms",
        f"trades.option.{currency}.100ms",
    ]
    if book_depth > 0:
        fixed.append(f"book.{perp}.none.{book_depth}.100ms")
    if option_book_depth > 0:
        fixed.extend(_book_channel(name, option_book_depth) for name in option_books)
    return fixed + [f"ticker.{name}.100ms" for name in option_names]


def save_instruments(root: Path, rows: list[dict], now_s: float) -> Path:
    stamp = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime(now_s))
    date = stamp[:10]
    directory = root / "instruments" / f"date={date}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"instruments-{stamp}.json.gz"
    temp = directory / f".tmp-{path.name}"
    temp.write_bytes(gzip.compress(json.dumps(rows, separators=(",", ":")).encode()))
    temp.replace(path)
    return path


class Capture:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.out).expanduser()
        self.sink = ParquetSink(
            self.root,
            flush_seconds=args.flush_seconds,
            min_free_gb=args.min_free_gb,
        )
        self.router = Router(self.sink, state_interval_ms=int(args.state_interval * 1000))
        self.stop = asyncio.Event()
        self._ids = 0
        self._subscribed: set[str] = set()
        self._pending_subscribe: dict[int, list[str]] = {}
        self._saved_names: frozenset[str] = frozenset()
        self._rows: list[dict] = []
        self._book_names: list[str] = []

    def _event(self, event: str, detail: str = "") -> None:
        self.sink.add("sessions", {"ts_ms": int(time.time() * 1000), "event": event, "detail": detail})

    async def _rpc(self, ws, method: str, params: dict) -> int:
        self._ids += 1
        await ws.send(
            json.dumps({"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params})
        )
        return self._ids

    async def _universe(self) -> list[str]:
        rows = await asyncio.to_thread(fetch_instruments, self.args.currency)
        self._rows = rows
        names = sorted(
            name
            for r in rows
            if r.get("is_active", True)
            and isinstance(name := r.get("instrument_name"), str)
            and name
        )
        if frozenset(names) != self._saved_names:
            path = await asyncio.to_thread(save_instruments, self.root, rows, time.time())
            self._saved_names = frozenset(names)
            _log(f"instrument list changed: {len(names)} options, saved {path.name}")
        return names

    async def _refresh_books(self) -> None:
        """Recompute which options get an order-book channel. A REST failure keeps the previous choice."""
        if self.args.option_book_depth <= 0:
            return
        try:
            summaries = await asyncio.to_thread(fetch_book_summaries, self.args.currency)
        except (OSError, TimeoutError, ValueError) as exc:
            _log(f"book summary unavailable, keeping {len(self._book_names)} option books: {exc}")
            return
        chosen = select_book_options(
            self._rows,
            summaries,
            int(time.time() * 1000),
            max_days=self.args.option_book_days,
            band=self.args.option_book_band,
            top_n=self.args.option_book_max,
        )
        if set(chosen) != set(self._book_names):
            _log(f"option books: {len(chosen)} instruments selected (was {len(self._book_names)})")
        self._book_names = chosen

    async def _drop_stale(self, ws, names: list[str]) -> None:
        stale = stale_channels(self._subscribed, self._channels(names))
        for i in range(0, len(stale), _CHUNK):
            chunk = stale[i : i + _CHUNK]
            await self._rpc(ws, "public/unsubscribe", {"channels": chunk})
            self._subscribed.difference_update(chunk)

    def _channels(self, names: list[str]) -> list[str]:
        return channels_for(
            names,
            self.args.currency,
            book_depth=self.args.book_depth,
            option_books=self._book_names,
            option_book_depth=self.args.option_book_depth,
        )

    async def _subscribe(self, ws, channels: list[str]) -> None:
        new = [c for c in channels if c not in self._subscribed]
        for i in range(0, len(new), _CHUNK):
            chunk = new[i : i + _CHUNK]
            req_id = await self._rpc(ws, "public/subscribe", {"channels": chunk})
            self._pending_subscribe[req_id] = chunk

    def _ack_subscribe(self, message: dict) -> bool:
        """Record a subscribe only after Deribit accepts it. Returns True if this frame was that reply."""
        req_id = message.get("id")
        if req_id not in self._pending_subscribe:
            return False
        channels = self._pending_subscribe.pop(req_id)
        if "error" in message:
            _log(f"subscribe rejected: {message['error']!r}")
            self._event("rpc_error", repr(message["error"])[:300])
            return True
        self._subscribed.update(channels)
        return True

    async def _session(self) -> None:
        import websockets

        names = await self._universe()
        await self._refresh_books()
        self._subscribed.clear()
        self._pending_subscribe.clear()
        async with websockets.connect(
            WS_URL, ping_interval=20, ping_timeout=20, open_timeout=30, max_size=None
        ) as ws:
            self._event("connect", f"options={len(names)}")
            await self._rpc(ws, "public/set_heartbeat", {"interval": 30})
            await self._subscribe(ws, self._channels(names))
            last_data = time.monotonic()
            last_refresh = time.monotonic()
            last_stats = time.monotonic()
            count = 0
            while not self.stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except TimeoutError:
                    raw = None
                now = time.monotonic()
                if raw is not None:
                    recv_ns = time.time_ns()
                    message = json.loads(raw if isinstance(raw, str) else raw.decode())
                    if self._ack_subscribe(message):
                        pass
                    elif "error" in message:
                        _log(f"rpc error: {message['error']!r}")
                        self._event("rpc_error", repr(message["error"])[:300])
                    elif message.get("method") == "heartbeat":
                        if (message.get("params") or {}).get("type") == "test_request":
                            await self._rpc(ws, "public/test", {})
                    else:
                        last_data = now
                        self.router.handle(message, recv_ns)
                        count += 1
                elif now - last_data > _SILENCE_S:
                    raise TimeoutError(f"no market data for {_SILENCE_S:.0f}s")
                self.sink.maybe_flush()
                if now - last_refresh >= self.args.refresh_seconds:
                    last_refresh = now
                    fresh = await self._universe()
                    await self._refresh_books()
                    await self._subscribe(ws, self._channels(fresh))
                    await self._drop_stale(ws, fresh)
                if now - last_stats >= self.args.stats_seconds:
                    elapsed = now - last_stats
                    last_stats = now
                    _log(
                        f"frames/s={count / elapsed:.0f} rows={dict(self.router.counts)} "
                        f"buffered={self.sink.buffered_rows} written={self.sink.rows_written} "
                        f"mb={self.sink.bytes_written / 1e6:.0f}"
                    )
                    count = 0

    async def run(self) -> int:
        import websockets

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop.set)
        backoff = 1.0
        exit_code = 0
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    await self._session()
                except asyncio.CancelledError:
                    raise
                except DiskLow as exc:
                    _log(f"STOP disk guard: {exc}")
                    exit_code = 2
                    break
                except (
                    OSError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    websockets.exceptions.WebSocketException,
                ) as exc:
                    _log(f"disconnect {type(exc).__name__}: {exc}")
                    self._event("disconnect", f"{type(exc).__name__}: {exc}"[:300])
                    if time.monotonic() - started > 60:
                        backoff = 1.0
                    try:
                        await asyncio.wait_for(self.stop.wait(), timeout=backoff)
                    except TimeoutError:
                        pass
                    backoff = min(backoff * 2.0, 30.0)
                else:
                    self._event("disconnect", "clean")
        finally:
            try:
                self._event("stop", "exit")
                self.sink.flush_all()
            except DiskLow as exc:
                _log(f"final flush failed: {exc}")
                exit_code = 2
            _log(f"stopped; rows written {self.sink.rows_written}")
        return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture Deribit option, perpetual and index ticks to Parquet.")
    p.add_argument("--out", required=True, help="output root directory")
    p.add_argument("--currency", default="BTC")
    p.add_argument("--state-interval", type=float, default=10.0, help="seconds between option_state rows per instrument")
    p.add_argument("--book-depth", type=int, default=10, help="perpetual book levels (0 disables)")
    p.add_argument("--option-book-depth", type=int, default=0, help="option book levels (0 disables option books)")
    p.add_argument("--option-book-max", type=int, default=200, help="max options with a book channel")
    p.add_argument("--option-book-days", type=float, default=35.0, help="only options expiring within this many days")
    p.add_argument("--option-book-band", type=float, default=0.15, help="max |ln(strike/forward)| for an option book")
    p.add_argument("--flush-seconds", type=float, default=60.0)
    p.add_argument("--refresh-seconds", type=float, default=600.0, help="how often to look for newly listed options")
    p.add_argument("--stats-seconds", type=float, default=60.0)
    p.add_argument("--min-free-gb", type=float, default=15.0, help="stop capturing below this much free disk")
    p.add_argument("--duration", type=float, default=None, help="stop after this many seconds (smoke tests)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    capture = Capture(args)

    async def runner() -> int:
        if args.duration is not None:
            asyncio.get_running_loop().call_later(args.duration, capture.stop.set)
        return await capture.run()

    return asyncio.run(runner())

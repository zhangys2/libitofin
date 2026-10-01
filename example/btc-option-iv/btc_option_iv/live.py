"""Notebook feed controller: event-time filters and detached three-second views."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict

from websockets.exceptions import WebSocketException

from .kalman import FilterConfig, FilterManager
from .market import (
    MISSING_FORWARD,
    MISSING_INDEX,
    STALE_AFTER_MS,
    FutureQuote,
    IndexQuote,
    MarketBook,
    Rejection,
    evaluate_quote,
)
from .stream import fetch_active_btc_options, subscribe_btc_option_market_data


class LiveFeedController:
    """Own the public feed and all expiry states, independently of UI selection."""

    def __init__(self, publish):
        self.publish = publish
        self.task = None
        self.filters = FilterManager()
        self.snapshot = {
            "status": "Feed stopped.",
            "latest_options": {},
            "instruments": {},
            "rows": {},
            "book": MarketBook(),
            "kalman_views": {},
            "filter_config": self.filters.config,
            "feed_running": False,
        }

    def _publish_snapshot(self, latest_options, instruments, rows, book, status):
        now_ms, now = int(time.time() * 1000), time.monotonic()
        running = self.task is not None and not self.task.done()
        if running:
            self.filters.tick(now_ms=now_ms, now=now)
        self.snapshot = {
            "status": status,
            "latest_options": dict(latest_options),
            "instruments": dict(instruments),
            "rows": dict(rows),
            "book": MarketBook(
                index_price=book.index_price,
                index_timestamp_ms=book.index_timestamp_ms,
                futures=dict(book.futures),
            ),
            "kalman_views": self.filters.snapshots(now=now),
            "filter_config": self.filters.config,
            "published_ms": now_ms,
            "feed_running": running,
        }
        self.publish(self.snapshot)

    def _republish(self):
        s = self.snapshot
        self._publish_snapshot(
            s["latest_options"], s["instruments"], s["rows"], s["book"], s["status"]
        )

    def configure(self, config: FilterConfig):
        running = self.task is not None and not self.task.done()
        if running:
            self.filters.configure(config, now=time.monotonic())
        else:
            self.filters.config = config
        self._republish()

    def reset_filters(self):
        self.filters.reset_all(now_ms=int(time.time() * 1000), now=time.monotonic())
        self._republish()

    async def set_enabled(self, enabled: bool):
        if enabled:
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(
                    self._run(), name="btc-option-iv-live-surface"
                )
            return
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
        self.snapshot = {
            **self.snapshot,
            "status": "Feed stopped — view frozen until resumed.",
        }
        self._republish()

    async def _run(self):
        rows, latest_options, by_name = {}, {}, {}
        book = MarketBook()
        backoff = 1.0

        async def publish_periodically():
            while True:
                self._publish_snapshot(
                    latest_options,
                    by_name,
                    rows,
                    book,
                    f"Streaming {len(by_name)} active option instruments.",
                )
                await asyncio.sleep(3.0)

        try:
            while True:
                try:
                    instruments, rejections = await asyncio.to_thread(
                        fetch_active_btc_options
                    )
                    by_name = {item.instrument_name: item for item in instruments}
                    self.filters.retire(
                        {item.expiration_timestamp_ms for item in instruments},
                        now_ms=int(time.time() * 1000),
                    )
                    latest_options = {
                        name: q for name, q in latest_options.items() if name in by_name
                    }
                    rows = {name: row for name, row in rows.items() if name in by_name}
                    by_future = defaultdict(list)
                    for item in instruments:
                        if item.future_name is not None:
                            by_future[item.future_name].append(item.instrument_name)
                    book = MarketBook()
                    processed = {}
                    self._publish_snapshot(
                        latest_options,
                        by_name,
                        rows,
                        book,
                        f"Connecting to {len(instruments)} options; {len(rejections)} rejected.",
                    )
                    publisher = asyncio.create_task(publish_periodically())
                    try:
                        async for event in subscribe_btc_option_market_data(
                            instruments, refresh_seconds=60.0
                        ):
                            if isinstance(event, IndexQuote):
                                book.note_index(event)
                                candidates = [
                                    name
                                    for name, quote in latest_options.items()
                                    if quote.index_price is None
                                ]
                            elif isinstance(event, FutureQuote):
                                book.note_future(event)
                                candidates = [
                                    name
                                    for name in by_future.get(event.instrument_name, [])
                                    if latest_options.get(name) is not None
                                    and latest_options[name].underlying_price is None
                                ]
                            else:
                                previous = latest_options.get(event.instrument_name)
                                if (
                                    previous is not None
                                    and event.timestamp_ms < previous.timestamp_ms
                                ):
                                    continue
                                latest_options[event.instrument_name] = event
                                candidates = [event.instrument_name]
                            for name in candidates:
                                quote, instrument = (
                                    latest_options.get(name),
                                    by_name.get(name),
                                )
                                if quote is None or instrument is None:
                                    continue
                                forwards = book.forward_data(instrument)
                                signature = (
                                    quote,
                                    quote.timestamp_ms
                                    if quote.index_price is not None
                                    else forwards.index_timestamp_ms,
                                    quote.timestamp_ms
                                    if quote.underlying_price is not None
                                    else forwards.forward_timestamp_ms,
                                )
                                if processed.get(name) == signature:
                                    continue
                                now_ms = int(time.time() * 1000)
                                outcome = evaluate_quote(
                                    quote,
                                    instrument,
                                    forwards,
                                    as_of_ms=now_ms,
                                    stale_after_ms=STALE_AFTER_MS,
                                )
                                if isinstance(
                                    outcome, Rejection
                                ) and outcome.reason in (
                                    MISSING_INDEX,
                                    MISSING_FORWARD,
                                ):
                                    continue
                                processed[name] = signature
                                if (
                                    isinstance(outcome, Rejection)
                                    or outcome.mid_iv is None
                                ):
                                    rows.pop(name, None)
                                    self.filters.discard(
                                        name, instrument.expiration_timestamp_ms
                                    )
                                else:
                                    rows[name] = outcome
                                    # Every source event, not just three-second snapshots.
                                    # The manager deduplicates premium content, so fallback
                                    # revaluations and heartbeats cannot add fake evidence.
                                    self.filters.ingest(
                                        outcome,
                                        instrument,
                                        now_ms=now_ms,
                                        now=time.monotonic(),
                                    )
                    finally:
                        publisher.cancel()
                        try:
                            await publisher
                        except asyncio.CancelledError:
                            pass
                    backoff = 1.0
                except (
                    OSError,
                    TimeoutError,
                    WebSocketException,
                    json.JSONDecodeError,
                ) as exc:
                    self._publish_snapshot(
                        latest_options,
                        by_name,
                        rows,
                        book,
                        f"Feed reconnecting after {type(exc).__name__}: {exc}",
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)
        except asyncio.CancelledError:
            self._publish_snapshot(
                latest_options,
                by_name,
                rows,
                book,
                "Feed stopped — view frozen until resumed.",
            )
            raise

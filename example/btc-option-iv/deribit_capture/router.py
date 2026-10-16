"""Turn Deribit subscription messages into sink rows. No network, no clock reads."""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

from deribit_capture.sink import ParquetSink

_OPTION_NAME = re.compile(r"^[A-Z]+-\d{1,2}[A-Z]{3}\d{2}-\d+(?:d\d+)?-[CP]$")
_TRADE_KNOWN = {
    "timestamp",
    "trade_id",
    "trade_seq",
    "price",
    "amount",
    "direction",
    "index_price",
    "mark_price",
    "iv",
    "tick_direction",
    "liquidation",
    "block_trade_id",
    "instrument_name",
}
_SEEN_TRADES = 50_000


def is_option(name: str) -> bool:
    return _OPTION_NAME.match(name) is not None


def _f(value: Any) -> float | None:
    """Number or None. Deribit sends numbers; anything else is treated as missing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class Router:
    """Dispatch one decoded WebSocket frame at a time.

    ``quotes`` rows are written only when top-of-book (price or size on either
    side) changes. ``option_state`` rows carry the slower fields (mark, Greeks,
    open interest) and are sampled at most once per ``state_interval_ms`` per
    instrument, because mark and Greeks move with every index tick and would
    otherwise dominate storage.
    """

    def __init__(self, sink: ParquetSink, *, state_interval_ms: int = 10_000) -> None:
        self.sink = sink
        self.state_interval_ms = state_interval_ms
        self._last_l1: dict[str, tuple] = {}
        self._last_option_book: dict[str, tuple] = {}
        self._last_state_ts: dict[str, int] = {}
        self._seen_trades: set[tuple[str, str]] = set()
        self._seen_order: deque[tuple[str, str]] = deque()
        self.counts: dict[str, int] = {}
        self.last_message_ts_ms: int | None = None

    def handle(self, message: dict, recv_ns: int) -> None:
        if message.get("method") != "subscription":
            return
        params = message.get("params")
        if not isinstance(params, dict):
            return
        channel = params.get("channel")
        data = params.get("data")
        if not isinstance(channel, str) or data is None:
            return
        if channel.startswith("ticker.") and isinstance(data, dict):
            self._ticker(data, recv_ns)
        elif channel.startswith("trades.") and isinstance(data, list):
            self._trades(data, recv_ns)
        elif channel.startswith("book.") and isinstance(data, dict):
            self._book(data, recv_ns)
        elif channel.startswith("deribit_price_index.") and isinstance(data, dict):
            self._index(data, recv_ns)

    def _count(self, table: str) -> None:
        self.counts[table] = self.counts.get(table, 0) + 1

    def _ticker(self, d: dict, recv_ns: int) -> None:
        name = d.get("instrument_name")
        ts = d.get("timestamp")
        if not isinstance(name, str) or not isinstance(ts, int):
            return
        self.last_message_ts_ms = ts
        if not is_option(name):
            row = {
                "ts_ms": ts,
                "recv_ns": recv_ns,
                "instrument": name,
                "best_bid_px": _f(d.get("best_bid_price")),
                "best_bid_amt": _f(d.get("best_bid_amount")),
                "best_ask_px": _f(d.get("best_ask_price")),
                "best_ask_amt": _f(d.get("best_ask_amount")),
                "mark_px": _f(d.get("mark_price")),
                "index_px": _f(d.get("index_price")),
                "last_px": _f(d.get("last_price")),
                "open_interest": _f(d.get("open_interest")),
                "funding_8h": _f(d.get("funding_8h")),
                "current_funding": _f(d.get("current_funding")),
                "state": d.get("state"),
            }
            self.sink.add("ticker_other", row)
            self._count("ticker_other")
            return

        l1 = (
            d.get("best_bid_price"),
            d.get("best_bid_amount"),
            d.get("best_ask_price"),
            d.get("best_ask_amount"),
        )
        if self._last_l1.get(name) != l1:
            self._last_l1[name] = l1
            self.sink.add(
                "quotes",
                {
                    "ts_ms": ts,
                    "recv_ns": recv_ns,
                    "instrument": name,
                    "bid_px": _f(l1[0]),
                    "bid_amt": _f(l1[1]),
                    "ask_px": _f(l1[2]),
                    "ask_amt": _f(l1[3]),
                    "bid_iv": _f(d.get("bid_iv")),
                    "ask_iv": _f(d.get("ask_iv")),
                    "mark_iv": _f(d.get("mark_iv")),
                    "mark_px": _f(d.get("mark_price")),
                    "index_px": _f(d.get("index_price")),
                    "underlying_px": _f(d.get("underlying_price")),
                },
            )
            self._count("quotes")

        last = self._last_state_ts.get(name)
        if last is None or ts - last >= self.state_interval_ms:
            self._last_state_ts[name] = ts
            greeks = d.get("greeks") if isinstance(d.get("greeks"), dict) else {}
            stats = d.get("stats") if isinstance(d.get("stats"), dict) else {}
            self.sink.add(
                "option_state",
                {
                    "ts_ms": ts,
                    "recv_ns": recv_ns,
                    "instrument": name,
                    "mark_px": _f(d.get("mark_price")),
                    "mark_iv": _f(d.get("mark_iv")),
                    "bid_iv": _f(d.get("bid_iv")),
                    "ask_iv": _f(d.get("ask_iv")),
                    "delta": _f(greeks.get("delta")),
                    "gamma": _f(greeks.get("gamma")),
                    "vega": _f(greeks.get("vega")),
                    "theta": _f(greeks.get("theta")),
                    "rho": _f(greeks.get("rho")),
                    "open_interest": _f(d.get("open_interest")),
                    "last_px": _f(d.get("last_price")),
                    "volume": _f(stats.get("volume")),
                    "volume_usd": _f(stats.get("volume_usd")),
                    "index_px": _f(d.get("index_price")),
                    "underlying_px": _f(d.get("underlying_price")),
                    "underlying_index": d.get("underlying_index"),
                    "state": d.get("state"),
                },
            )
            self._count("option_state")

    def _trades(self, trades: list, recv_ns: int) -> None:
        for t in trades:
            if not isinstance(t, dict):
                continue
            name = t.get("instrument_name")
            ts = t.get("timestamp")
            trade_id = t.get("trade_id")
            if not isinstance(name, str) or not isinstance(ts, int):
                continue
            key = (name, str(trade_id))
            if key in self._seen_trades:
                continue
            self._seen_trades.add(key)
            self._seen_order.append(key)
            if len(self._seen_order) > _SEEN_TRADES:
                self._seen_trades.discard(self._seen_order.popleft())
            extra = {k: v for k, v in t.items() if k not in _TRADE_KNOWN}
            tick = t.get("tick_direction")
            self.sink.add(
                "trades",
                {
                    "ts_ms": ts,
                    "recv_ns": recv_ns,
                    "instrument": name,
                    "trade_id": None if trade_id is None else str(trade_id),
                    "trade_seq": t.get("trade_seq") if isinstance(t.get("trade_seq"), int) else None,
                    "price": _f(t.get("price")),
                    "amount": _f(t.get("amount")),
                    "direction": t.get("direction"),
                    "index_px": _f(t.get("index_price")),
                    "mark_px": _f(t.get("mark_price")),
                    "iv": _f(t.get("iv")),
                    "tick_direction": tick if isinstance(tick, int) else None,
                    "liquidation": t.get("liquidation"),
                    "block_trade_id": None
                    if t.get("block_trade_id") is None
                    else str(t["block_trade_id"]),
                    "extra_json": json.dumps(extra, separators=(",", ":")) if extra else None,
                },
            )
            self._count("trades")

    def _book(self, d: dict, recv_ns: int) -> None:
        ts = d.get("timestamp")
        name = d.get("instrument_name")
        bids = d.get("bids")
        asks = d.get("asks")
        if not isinstance(ts, int) or not isinstance(name, str):
            return
        if not isinstance(bids, list) or not isinstance(asks, list):
            return

        def side(levels: list) -> tuple[list[float], list[float]]:
            # Snapshot-style book frames carry [price, amount]. Delta-style frames carry
            # [action, price, amount]; those are rejected rather than mis-parsed.
            prices: list[float] = []
            amounts: list[float] = []
            for level in levels:
                if (
                    isinstance(level, list)
                    and len(level) == 2
                    and all(isinstance(x, (int, float)) for x in level)
                ):
                    prices.append(float(level[0]))
                    amounts.append(float(level[1]))
            return prices, amounts

        bid_px, bid_amt = side(bids)
        ask_px, ask_amt = side(asks)
        if len(bid_px) != len(bids) or len(ask_px) != len(asks):
            return
        table = "option_book" if is_option(name) else "perp_book"
        if table == "option_book":
            # Snapshots arrive on a timer; store an option's book only when its content changes.
            content = (tuple(bid_px), tuple(bid_amt), tuple(ask_px), tuple(ask_amt))
            if self._last_option_book.get(name) == content:
                return
            self._last_option_book[name] = content
        self.sink.add(
            table,
            {
                "ts_ms": ts,
                "recv_ns": recv_ns,
                "change_id": d.get("change_id") if isinstance(d.get("change_id"), int) else None,
                "instrument": name,
                "bid_px": bid_px,
                "bid_amt": bid_amt,
                "ask_px": ask_px,
                "ask_amt": ask_amt,
            },
        )
        self._count(table)

    def _index(self, d: dict, recv_ns: int) -> None:
        ts = d.get("timestamp")
        price = _f(d.get("price"))
        if not isinstance(ts, int) or price is None:
            return
        self.sink.add(
            "index",
            {"ts_ms": ts, "recv_ns": recv_ns, "index_name": d.get("index_name"), "price": price},
        )
        self._count("index")

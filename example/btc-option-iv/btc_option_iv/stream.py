"""Deribit public market-data client for the BTC option IV example.

Networking stays in this module. ``itofin`` only sees normalized Black inputs.
"""

from __future__ import annotations

# standard library
import argparse
import asyncio
import json
import sys
import time
import urllib.request
from collections.abc import AsyncIterator

from btc_option_iv.market import (
    STALE_AFTER_MS,
    FutureQuote,
    IndexQuote,
    Instrument,
    MarketBook,
    OptionQuote,
    Rejection,
    evaluate_quote,
    format_iv_row,
    format_rejection,
    options_with_listed_futures,
    parse_future_instruments,
    parse_option_instruments,
    parse_subscription_message,
    subscription_channels,
)

WS_URL = "wss://www.deribit.com/ws/api/v2"
_OPTIONS_URL = (
    "https://www.deribit.com/api/v2/public/get_instruments"
    "?currency=BTC&kind=option&expired=false"
)
_FUTURES_URL = (
    "https://www.deribit.com/api/v2/public/get_instruments"
    "?currency=BTC&kind=future&expired=false"
)
_CHUNK = 100


def fetch_active_btc_options(
    now_ms: int | None = None,
) -> tuple[list[Instrument], list[Rejection]]:
    """Load active inverse BTC options that have a listed dated future."""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    options, rejections = parse_option_instruments(_get_json(_OPTIONS_URL), now_ms=now)
    futures = parse_future_instruments(_get_json(_FUTURES_URL), now_ms=now)
    kept, missing = options_with_listed_futures(options, futures)
    return kept, rejections + missing


async def subscribe_btc_option_market_data(
    instruments: list[Instrument],
    *,
    refresh_seconds: float = 60.0,
) -> AsyncIterator[OptionQuote | FutureQuote | IndexQuote]:
    """Subscribe to the public ticker and index channels until the refresh interval elapses."""
    import websockets

    channels = subscription_channels(instruments)
    async with websockets.connect(
        WS_URL, ping_interval=20, ping_timeout=20, open_timeout=30
    ) as ws:
        ids = [0]
        await _rpc(ws, ids, "public/set_heartbeat", {"interval": 30})
        for start in range(0, len(channels), _CHUNK):
            await _rpc(
                ws,
                ids,
                "public/subscribe",
                {"channels": channels[start : start + _CHUNK]},
            )
        deadline = time.monotonic() + refresh_seconds
        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                return
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except TimeoutError:
                return
            payload = json.loads(raw if isinstance(raw, str) else raw.decode())
            if "error" in payload:
                print(f"REJECT deribit reason=rpc_error detail={payload['error']!r}", file=sys.stderr)
                continue
            if payload.get("method") == "heartbeat":
                if (payload.get("params") or {}).get("type") == "test_request":
                    await _rpc(ws, ids, "public/test", {})
                continue
            event = parse_subscription_message(payload)
            if event is not None:
                yield event


async def run_live_btc_iv_stream(
    output=None,
    *,
    refresh_seconds: float = 60.0,
    stale_after_ms: int = STALE_AFTER_MS,
    max_rows: int | None = None,
) -> None:
    """Print timestamped bid/ask/mid IV rows until interrupted or ``max_rows`` is reached."""
    sink = sys.stdout if output is None else output
    emitted = 0
    backoff = 1.0
    while True:
        try:
            instruments, rejections = await asyncio.to_thread(fetch_active_btc_options)
            for rejection in rejections:
                print(format_rejection(rejection), file=sys.stderr)
            by_name = {item.instrument_name: item for item in instruments}
            book = MarketBook()
            async for event in subscribe_btc_option_market_data(
                instruments, refresh_seconds=refresh_seconds
            ):
                if isinstance(event, IndexQuote):
                    book.note_index(event)
                    continue
                if isinstance(event, FutureQuote):
                    book.note_future(event)
                    continue
                instrument = by_name.get(event.instrument_name)
                if instrument is None:
                    continue
                outcome = evaluate_quote(
                    event,
                    instrument,
                    book.forward_data(instrument),
                    as_of_ms=int(time.time() * 1000),
                    stale_after_ms=stale_after_ms,
                )
                if isinstance(outcome, Rejection):
                    print(format_rejection(outcome), file=sys.stderr)
                    continue
                print(format_iv_row(outcome), file=sink, flush=True)
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return
            backoff = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"RECONNECT {type(exc).__name__}: {exc}", file=sys.stderr)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for a manual live smoke run."""
    parser = argparse.ArgumentParser(
        description="Stream Deribit BTC option Black implied volatilities."
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--stale-after-ms", type=int, default=STALE_AFTER_MS)
    parser.add_argument("--refresh-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    try:
        asyncio.run(
            run_live_btc_iv_stream(
                refresh_seconds=args.refresh_seconds,
                stale_after_ms=args.stale_after_ms,
                max_rows=args.max_rows,
            )
        )
    except KeyboardInterrupt:
        return


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "itofin-btc-option-iv"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("Deribit response is not an object")
    return payload


async def _rpc(ws, ids: list[int], method: str, params: dict) -> None:
    ids[0] += 1
    await ws.send(
        json.dumps({"jsonrpc": "2.0", "id": ids[0], "method": method, "params": params})
    )

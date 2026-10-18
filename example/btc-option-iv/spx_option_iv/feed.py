"""SPX 0DTE live feed controller with IBKR socket and Bayesian Kalman filter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import statistics
import time
from zoneinfo import ZoneInfo

from itofin import ItofinError, implied_forward
from itofin.instruments import OptionType
from itofin.pricingengines import black_formula_implied_volatility

from btc_option_iv.kalman import FilterConfig, FilterManager


# Stop pricing this long before the close: below it Black-76 IVs of 0DTE
# options are dominated by noise and the Kalman knot window collapses.
MIN_SECONDS_TO_EXPIRY = 30.0
YEAR_MS = 365.25 * 86_400_000
# Put-call parity forward: use calls AND puts at the strikes nearest the money.
PARITY_STRIKES = 7  # strikes (nearest spot) subscribed on both rights
PARITY_MAX_PAIRS = 5  # tightest pairs combined by median
PARITY_MIN_PAIRS = 2
PARITY_MAX_DEVIATION = 0.02  # reject a parity forward >2% away from spot
MARKET_DATA_TYPES = {
    1: "Live",
    2: "Frozen",
    3: "Delayed (~15 min)",
    4: "Delayed frozen",
}


def _load_ib():
    """Import ib_async lazily, inside the running loop that will own the sockets.

    Importing at module level would make ib_async/eventkit bind to whatever loop
    exists at import time (a different one from marimo's async cells), and would
    make ``pytest`` collection require the optional notebook dependency.
    """
    from ib_async import IB, Index, Option, Stock

    return IB, Index, Option, Stock


@dataclass(frozen=True)
class SpxInstrumentMeta:
    instrument_name: str
    expiration_timestamp_ms: int
    option_type: str
    strike: float


@dataclass(frozen=True)
class SpxIvRow:
    instrument_name: str
    timestamp_ms: int
    strike: float
    mid_iv: float
    bid_iv: float
    ask_iv: float
    forward: float
    expiry: float
    # SPX quotes are index points, not BTC or USD cash. The premium_btc names
    # only match the FilterManager fingerprint; *_usd are the same points (the
    # views label them "usd" for table-layout parity with the BTC reader).
    bid_premium_btc: float
    ask_premium_btc: float
    bid_usd: float
    ask_usd: float
    right: str


def _finite_positive(value) -> float | None:
    """Return ``value`` as a float if finite and positive, else None (NaN is truthy)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _implied_vol(opt_type, strike, forward, expiry, price) -> float | None:
    """Black-76 IV for ``price``, or None when it cannot be inverted."""
    try:
        iv = black_formula_implied_volatility(
            opt_type,
            strike=strike,
            forward=forward,
            expiry=expiry,
            black_price=price,
            discount=1.0,
        )
    except (ItofinError, ValueError, OverflowError):
        return None
    return iv if math.isfinite(iv) and iv > 0 else None


def _two_sided(ticker) -> tuple[float, float] | None:
    """(bid, ask) when the ticker has a valid two-sided quote, else None."""
    bid, ask = _finite_positive(ticker.bid), _finite_positive(ticker.ask)
    return (bid, ask) if bid is not None and ask is not None and ask >= bid else None


def parity_forward(
    pairs, spot: float, expiry_years: float = 0.01
) -> tuple[float, int] | None:
    """Forward implied by put-call parity using ``itofin.implied_forward``.

    ``pairs`` holds ``(strike, call_bid, call_ask, put_bid, put_ask)`` in index
    points. Uses itofin's unified weighted least squares and arbitrage diagnostic
    engine. Returns (forward, pairs_used) or None if parity cannot be established.
    """
    pairs_list = [
        p
        for p in pairs
        if 0 < (p[1] + p[2]) * 0.5 < 0.1 * spot and 0 < (p[3] + p[4]) * 0.5 < 0.1 * spot
    ]
    if len(pairs_list) < PARITY_MIN_PAIRS:
        return None

    try:
        res = implied_forward(
            quotes=pairs_list,
            expiry_years=max(expiry_years, 1e-4),
            convention="european",
            spot=spot,
            max_spot_deviation=PARITY_MAX_DEVIATION,
            discount_factor=1.0,
            min_pairs=PARITY_MIN_PAIRS,
            max_pairs=PARITY_MAX_PAIRS,
        )
    except Exception:
        return None

    if res is None:
        return None
    if not math.isfinite(res.forward) or res.forward <= 0:
        return None
    if abs(res.forward / spot - 1.0) > PARITY_MAX_DEVIATION:
        return None
    return res.forward, res.pairs_used


def _parity_pairs(quotes, strikes):
    """(strike, call_bid, call_ask, put_bid, put_ask) for strikes quoted on both rights."""
    for strike in strikes:
        call, put = quotes.get((strike, "C")), quotes.get((strike, "P"))
        if call is not None and put is not None:
            yield (strike, *call, *put)


def select_spxw_chain(chains):
    """SPXW chain on SMART (else any SPXW chain); never the monthly SPX chain."""
    spxw = [c for c in chains if c.tradingClass == "SPXW"]
    return next((c for c in spxw if c.exchange == "SMART"), spxw[0] if spxw else None)


def select_chain(chains, symbol: str = "SPX"):
    """Select the appropriate 0DTE-capable chain for SPX (SPXW) or SPY."""
    if symbol == "SPY":
        smart = [c for c in chains if c.exchange == "SMART" and c.tradingClass == "SPY"]
        return (
            smart[0]
            if smart
            else next((c for c in chains if c.tradingClass == "SPY"), None)
        )
    return select_spxw_chain(chains)


def select_live_expiry(expirations, now_ms: int) -> tuple[str, int] | None:
    """Earliest expiration whose 4:00 PM ET close is still in the future."""
    for expiry_str in sorted(expirations):
        expiry_ms = compute_0dte_expiry_ms(expiry_str)
        if expiry_ms > now_ms:
            return expiry_str, expiry_ms
    return None


def compute_0dte_expiry_ms(expiry_str: str) -> int:
    """Calculate 4:00 PM US Eastern Time market close timestamp for the contract date."""
    dt = datetime.strptime(expiry_str, "%Y%m%d")
    ny_tz = ZoneInfo("America/New_York")
    close_dt = dt.replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=ny_tz)
    return int(close_dt.timestamp() * 1000)


class SpxFeedController:
    """Owns the IBKR socket connection, 0DTE option chain subscriptions, and Kalman filter."""

    def __init__(
        self,
        publish,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 42,
        symbol: str = "SPX",
        **kwargs,
    ):
        self.publish = publish
        self.host = host
        self.port = port
        self.client_id = kwargs.get("client_id", client_id)
        self.symbol = symbol.upper()
        self.strike_radius = 8.0 if self.symbol == "SPY" else 80.0
        self.market_data_type = 1  # live; 3 = delayed (free) fallback
        self.ingest_errors = 0
        self.recorder = kwargs.get("recorder")
        self.task: asyncio.Task | None = None
        self.filters = FilterManager()
        self.snapshot = {
            "symbol": self.symbol,
            "status": "Feed stopped. Switch on to connect to IB Gateway.",
            "connected": False,
            "spot_price": None,
            "expiry_str": None,
            "expiry_timestamp_ms": None,
            "time_to_close_hours": None,
            "latest_quotes": {},
            "kalman_views": {},
            "filter_config": self.filters.config,
            "feed_running": False,
            "published_ms": int(time.time() * 1000),
        }
        self._desired_enabled = False
        self._lifecycle_lock = asyncio.Lock()

    def configure(self, config: FilterConfig):
        running = self.task is not None and not self.task.done()
        if running:
            self.filters.configure(config, now=time.monotonic())
        else:
            self.filters.config = config
        self._republish()

    def reset_filters(self):
        now_ms, now = int(time.time() * 1000), time.monotonic()
        self.filters.reset_all(now_ms=now_ms, now=now)
        self._republish()

    async def set_symbol(self, symbol: str):
        sym = symbol.upper()
        async with self._lifecycle_lock:
            if sym == self.symbol:
                return
            task = self.task
            was_running = task is not None and not task.done()
            if was_running:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                if self.task is task:
                    self.task = None
            self.symbol = sym
            self.strike_radius = 8.0 if sym == "SPY" else 80.0
            now_ms, now = int(time.time() * 1000), time.monotonic()
            self.filters.reset_all(now_ms=now_ms, now=now)
            should_restart = was_running and self._desired_enabled
            self.snapshot = {
                "symbol": self.symbol,
                "status": f"Switched to {self.symbol}. "
                + (
                    "Reconnecting..."
                    if should_restart
                    else "Switch on to connect to IB Gateway."
                ),
                "connected": False,
                "spot_price": None,
                "expiry_str": None,
                "expiry_timestamp_ms": None,
                "time_to_close_hours": None,
                "latest_quotes": {},
                "kalman_views": {},
                "filter_config": self.filters.config,
                "feed_running": should_restart,
                "published_ms": now_ms,
            }
            self.publish(self.snapshot)
            if should_restart:
                self.task = asyncio.create_task(
                    self._run(), name=f"{sym.lower()}-0dte-live-feed"
                )

    def set_port(self, port: int):
        self.port = port

    def set_market_data_type(self, market_data_type: int):
        if market_data_type not in MARKET_DATA_TYPES:
            raise ValueError(f"unknown market data type {market_data_type}")
        self.market_data_type = market_data_type

    def set_strike_radius(self, radius: float):
        self.strike_radius = radius

    def set_recorder(self, recorder):
        self.recorder = recorder

    def _republish(self):
        self.snapshot = {
            **self.snapshot,
            "kalman_views": self.filters.snapshots(now=time.monotonic()),
            "filter_config": self.filters.config,
            "published_ms": int(time.time() * 1000),
            "feed_running": self.task is not None and not self.task.done(),
        }
        self.publish(self.snapshot)

    async def set_enabled(self, enabled: bool):
        self._desired_enabled = enabled
        async with self._lifecycle_lock:
            if enabled:
                if self.task is None or self.task.done():
                    self.task = asyncio.create_task(
                        self._run(), name=f"{self.symbol.lower()}-0dte-live-feed"
                    )
                return
            task = self.task
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if self.task is task:
                self.task = None
            if self.recorder is not None:
                await self.recorder.flush_async()
            self.snapshot = {
                **self.snapshot,
                "status": "Feed stopped — view frozen until resumed.",
                "connected": False,
                "feed_running": False,
                "published_ms": int(time.time() * 1000),
            }
            self.publish(self.snapshot)

    def _status_prefix(self) -> str:
        label = MARKET_DATA_TYPES[self.market_data_type]
        return "" if self.market_data_type == 1 else f"[{label} data, not live] "

    async def _run(self):
        ib = None
        try:
            ib_items = _load_ib()
            IB, Index, Option = ib_items[:3]
            Stock = ib_items[3] if len(ib_items) > 3 else Index
            ib = IB()
            self.snapshot = {
                **self.snapshot,
                "symbol": self.symbol,
                "status": f"Connecting to IB Gateway on {self.host}:{self.port}...",
                "feed_running": True,
            }
            self.publish(self.snapshot)

            connected = False
            for cid in range(self.client_id, self.client_id + 15):
                try:
                    await ib.connectAsync(
                        self.host, self.port, clientId=cid, timeout=6, readonly=True
                    )
                    self.client_id = cid
                    connected = True
                    break
                except Exception as e:
                    if (
                        "already in use" in str(e).lower()
                        or "peer closed" in str(e).lower()
                    ):
                        continue
                    raise
            if not connected:
                raise RuntimeError("Unable to connect to IB Gateway (clientIds in use)")
            # 1 = live (needs an OPRA subscription), 3 = delayed (~15 min stale).
            ib.reqMarketDataType(self.market_data_type)

            # Qualify Underlying Contract
            if self.symbol == "SPY":
                underlying = Stock("SPY", "SMART", "USD")
                sec_type = "STK"
            else:
                underlying = Index("SPX", "CBOE")
                sec_type = "IND"

            await ib.qualifyContractsAsync(underlying)
            ticker_underlying = ib.reqMktData(underlying)

            # Wait for a finite spot price; marketPrice(), last, and close are NaN
            # (which is truthy) until data arrives.
            spot = None
            for attempt in range(10):
                await asyncio.sleep(1.0)
                spot = (
                    _finite_positive(ticker_underlying.marketPrice())
                    or _finite_positive(getattr(ticker_underlying, "last", None))
                    or _finite_positive(ticker_underlying.close)
                )
                if spot:
                    break
                # If requesting live data (1) but no spot arrives after 2 attempts,
                # fall back to delayed data (3) since accounts without an active OPRA subscription
                # will never receive a live SPX/SPY spot price.
                if attempt == 2 and self.market_data_type == 1:
                    self.market_data_type = 3
                    if hasattr(ib, "cancelMktData"):
                        ib.cancelMktData(underlying)
                    ib.reqMarketDataType(3)
                    ticker_underlying = ib.reqMktData(underlying)

            if not spot:
                raise RuntimeError(
                    f"No {self.symbol} spot price received from IBKR. Check that market data is enabled or gateway is open"
                )

            # Request Option Parameters
            chains = await ib.reqSecDefOptParamsAsync(
                self.symbol, "", sec_type, underlying.conId
            )
            chain = select_chain(chains, self.symbol)
            if not chain:
                msg = (
                    "No SPXW option chains found from IBKR"
                    if self.symbol == "SPX"
                    else f"No {self.symbol} option chains found from IBKR"
                )
                raise RuntimeError(msg)

            live_expiry = select_live_expiry(chain.expirations, int(time.time() * 1000))
            if live_expiry is None:
                raise RuntimeError(
                    f"No unexpired {self.symbol} expirations (market closed?)"
                )
            expiry_str, expiry_ms = live_expiry
            # Only the subscribed session may own a filter (leftovers from an
            # earlier connection would otherwise show up in the reader).
            self.filters.retire({expiry_ms}, now_ms=int(time.time() * 1000))

            # Select strikes around spot
            effective_radius = (
                self.strike_radius
                if (self.symbol != "SPY" or self.strike_radius <= 25.0)
                else max(5.0, round(self.strike_radius / 10.0))
            )
            if self.symbol == "SPY":
                strikes = [
                    s
                    for s in sorted(chain.strikes)
                    if abs(s - spot) <= effective_radius
                ]
            else:
                strikes = [
                    s
                    for s in sorted(chain.strikes)
                    if abs(s - spot) <= effective_radius and s % 5 == 0
                ]
            if not strikes:
                strikes = sorted(chain.strikes, key=lambda s: abs(s - spot))[:20]

            # Both rights at the strikes nearest the money (for put-call parity
            # and a tighter-spread choice at those strikes); the OTM right
            # elsewhere. The filter combines same-strike quotes itself.
            atm_strikes = set(
                sorted(strikes, key=lambda k: abs(k - spot))[:PARITY_STRIKES]
            )
            contracts = []
            for s in strikes:
                rights = (
                    ("C", "P") if s in atm_strikes else (("P" if s < spot else "C"),)
                )
                for right in rights:
                    contracts.append(
                        Option(
                            self.symbol,
                            expiry_str,
                            s,
                            right,
                            chain.exchange,
                            tradingClass=chain.tradingClass,
                        )
                    )

            await ib.qualifyContractsAsync(*contracts)
            tickers = [
                ib.reqMktData(c, genericTickList="100,101,106") for c in contracts
            ]

            self.snapshot = {
                **self.snapshot,
                "symbol": self.symbol,
                "status": (
                    f"{self._status_prefix()}Subscribed to {len(contracts)} "
                    f"{self.symbol} 0DTE contracts ({expiry_str})"
                ),
                "connected": True,
                "spot_price": spot,
                "expiry_str": expiry_str,
                "expiry_timestamp_ms": expiry_ms,
                "market_data_type": self.market_data_type,
            }
            self.publish(self.snapshot)

            # Live polling & calibration loop
            while True:
                await asyncio.sleep(2.0)
                now_ms, now = int(time.time() * 1000), time.monotonic()
                spot = (
                    _finite_positive(ticker_underlying.marketPrice())
                    or _finite_positive(getattr(ticker_underlying, "last", None))
                    or _finite_positive(ticker_underlying.close)
                    or spot
                )

                time_to_close_ms = expiry_ms - now_ms
                if time_to_close_ms <= MIN_SECONDS_TO_EXPIRY * 1000:
                    self.filters.retire(set(), now_ms=now_ms)
                    self.snapshot = {
                        **self.snapshot,
                        "symbol": self.symbol,
                        "status": (
                            f"{self.symbol} {expiry_str} session closed (4:00 PM ET) — "
                            "feed stopped, no quotes are being priced."
                        ),
                        "connected": True,
                        "latest_quotes": {},
                        "kalman_views": {},
                        "time_to_close_hours": 0.0,
                        "published_ms": now_ms,
                        "feed_running": False,
                    }
                    self.publish(self.snapshot)
                    return

                hours_left = time_to_close_ms / (3600 * 1000)
                # Actual remaining time, no floor: a floor would invert every IV
                # against a fake expiry (IV too small by sqrt(T_actual / T_floor)).
                t_years = time_to_close_ms / YEAR_MS

                # Forward from put-call parity on the near-ATM strikes; fall back
                # to spot (and say so) when there are too few clean pairs.
                quotes = {
                    (c.strike, c.right): _two_sided(t)
                    for c, t in zip(contracts, tickers)
                }
                parity = parity_forward(_parity_pairs(quotes, atm_strikes), spot)
                if parity is not None:
                    forward, forward_source = (
                        parity[0],
                        f"put-call parity ({parity[1]} strikes)",
                    )
                else:
                    forward, forward_source = spot, "spot (parity unavailable)"

                latest_quotes = {}

                for c, t in zip(contracts, tickers):
                    if quotes[(c.strike, c.right)] is not None:
                        if self.recorder is not None:
                            await self.recorder.record_quote_async(
                                symbol=self.symbol,
                                expiry=expiry_str,
                                strike=c.strike,
                                right=c.right,
                                bid=t.bid,
                                ask=t.ask,
                                bid_size=getattr(t, "bidSize", 0),
                                ask_size=getattr(t, "askSize", 0),
                                underlying_spot=spot,
                                timestamp_ns=now_ms * 1_000_000,
                            )
                        mid = (t.bid + t.ask) / 2.0
                        opt_type = OptionType.Put if c.right == "P" else OptionType.Call
                        # Invert bid/ask/mid with Black-76. A quote whose bid or ask
                        # cannot be inverted is skipped, never filled with a
                        # made-up spread (it would set the filter's precision).
                        bid_iv = _implied_vol(
                            opt_type, c.strike, forward, t_years, t.bid
                        )
                        ask_iv = _implied_vol(
                            opt_type, c.strike, forward, t_years, t.ask
                        )
                        if bid_iv is None or ask_iv is None:
                            continue
                        mid_iv = _implied_vol(opt_type, c.strike, forward, t_years, mid)
                        if mid_iv is None:
                            mid_iv = (bid_iv + ask_iv) / 2.0
                        if ask_iv < bid_iv:
                            bid_iv, ask_iv = ask_iv, bid_iv
                        mid_iv = max(bid_iv, min(ask_iv, mid_iv))

                        strike_str = (
                            int(c.strike) if c.strike == int(c.strike) else c.strike
                        )
                        name = f"{self.symbol}_{expiry_str}_{strike_str}_{c.right}"
                        inst = SpxInstrumentMeta(
                            instrument_name=name,
                            expiration_timestamp_ms=expiry_ms,
                            option_type=c.right.lower(),
                            strike=c.strike,
                        )
                        row = SpxIvRow(
                            instrument_name=name,
                            timestamp_ms=now_ms,
                            strike=c.strike,
                            mid_iv=mid_iv,
                            bid_iv=bid_iv,
                            ask_iv=ask_iv,
                            forward=forward,
                            expiry=t_years,
                            bid_premium_btc=t.bid,
                            ask_premium_btc=t.ask,
                            bid_usd=t.bid,
                            ask_usd=t.ask,
                            right=c.right,
                        )
                        try:
                            self.filters.ingest(row, inst, now_ms=now_ms, now=now)
                        except (ValueError, ItofinError):
                            # Expected per-quote rejections (crossed spread, bad
                            # inputs). Counted and shown; anything else is a bug
                            # and propagates to the error status below.
                            self.ingest_errors += 1
                            continue
                        latest_quotes[name] = row

                # Run Kalman prediction tick across active filters
                self.filters.tick(now_ms=now_ms, now=now)

                rejects = (
                    f", {self.ingest_errors} rejected" if self.ingest_errors else ""
                )
                self.snapshot = {
                    "symbol": self.symbol,
                    "status": (
                        f"{self._status_prefix()}Streaming {self.symbol} 0DTE quotes "
                        f"({len(latest_quotes)} active strikes{rejects})"
                    ),
                    "connected": True,
                    "spot_price": spot,
                    "forward": forward,
                    "forward_source": forward_source,
                    "expiry_str": expiry_str,
                    "expiry_timestamp_ms": expiry_ms,
                    "time_to_close_hours": hours_left,
                    "latest_quotes": latest_quotes,
                    "kalman_views": self.filters.snapshots(now=now),
                    "filter_config": self.filters.config,
                    "market_data_type": self.market_data_type,
                    "published_ms": now_ms,
                    "feed_running": True,
                }
                self.publish(self.snapshot)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.snapshot = {
                **self.snapshot,
                "status": f"IBKR Error: {exc}",
                "connected": False,
                "feed_running": False,
                "published_ms": int(time.time() * 1000),
            }
            self.publish(self.snapshot)
        finally:
            if ib is not None and ib.isConnected():
                ib.disconnect()

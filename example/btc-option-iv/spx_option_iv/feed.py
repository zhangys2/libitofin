"""SPX 0DTE live feed controller with IBKR socket and Bayesian Kalman filter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time
from zoneinfo import ZoneInfo

from itofin import ItofinError
from itofin.instruments import OptionType
from itofin.pricingengines import black_formula_implied_volatility

from btc_option_iv.kalman import FilterConfig, FilterManager


# Stop pricing this long before the close: below it Black-76 IVs of 0DTE
# options are dominated by noise and the Kalman knot window collapses.
MIN_SECONDS_TO_EXPIRY = 30.0
YEAR_MS = 365.25 * 86_400_000
MARKET_DATA_TYPES = {1: "Live", 2: "Frozen", 3: "Delayed (~15 min)", 4: "Delayed frozen"}


def _load_ib():
    """Import ib_async lazily, inside the running loop that will own the sockets.

    Importing at module level would make ib_async/eventkit bind to whatever loop
    exists at import time (a different one from marimo's async cells), and would
    make ``pytest`` collection require the optional notebook dependency.
    """
    from ib_async import IB, Index, Option

    return IB, Index, Option


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


def select_spxw_chain(chains):
    """SPXW chain on SMART (else any SPXW chain); never the monthly SPX chain."""
    spxw = [c for c in chains if c.tradingClass == "SPXW"]
    return next((c for c in spxw if c.exchange == "SMART"), spxw[0] if spxw else None)


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
        **kwargs,
    ):
        self.publish = publish
        self.host = host
        self.port = port
        self.client_id = kwargs.get("client_id", client_id)
        self.strike_radius = 80.0
        self.market_data_type = 1  # live; 3 = delayed (free) fallback
        self.ingest_errors = 0
        self.task: asyncio.Task | None = None
        self.filters = FilterManager()
        self.snapshot = {
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

    def set_port(self, port: int):
        self.port = port

    def set_market_data_type(self, market_data_type: int):
        if market_data_type not in MARKET_DATA_TYPES:
            raise ValueError(f"unknown market data type {market_data_type}")
        self.market_data_type = market_data_type

    def set_strike_radius(self, radius: float):
        self.strike_radius = radius

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
        if enabled:
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(self._run(), name="spx-0dte-live-feed")
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
            IB, Index, Option = _load_ib()
            ib = IB()
            self.snapshot = {
                **self.snapshot,
                "status": f"Connecting to IB Gateway on {self.host}:{self.port}...",
                "feed_running": True,
            }
            self.publish(self.snapshot)

            connected = False
            for cid in range(self.client_id, self.client_id + 15):
                try:
                    await ib.connectAsync(self.host, self.port, clientId=cid, timeout=6)
                    self.client_id = cid
                    connected = True
                    break
                except Exception as e:
                    if "already in use" in str(e).lower() or "peer closed" in str(e).lower():
                        continue
                    raise
            if not connected:
                raise RuntimeError("Unable to connect to IB Gateway (clientIds in use)")
            # 1 = live (needs an OPRA subscription), 3 = delayed (~15 min stale).
            ib.reqMarketDataType(self.market_data_type)

            # Qualify SPX Underlying Index
            spx = Index("SPX", "CBOE")
            await ib.qualifyContractsAsync(spx)
            ticker_spx = ib.reqMktData(spx)

            # Wait for a finite spot price; marketPrice() and close are NaN
            # (which is truthy) until data arrives.
            spot = None
            for _ in range(10):
                await asyncio.sleep(1.0)
                spot = _finite_positive(ticker_spx.marketPrice()) or _finite_positive(
                    ticker_spx.close
                )
                if spot:
                    break
            if not spot:
                raise RuntimeError("No SPX spot price received from IBKR")

            # Request SPX Option Parameters
            chains = await ib.reqSecDefOptParamsAsync("SPX", "", "IND", spx.conId)
            chain = select_spxw_chain(chains)
            if not chain:
                raise RuntimeError("No SPXW option chains found from IBKR")

            live_expiry = select_live_expiry(chain.expirations, int(time.time() * 1000))
            if live_expiry is None:
                raise RuntimeError("No unexpired SPXW expirations (market closed?)")
            expiry_str, expiry_ms = live_expiry
            # Only the subscribed session may own a filter (leftovers from an
            # earlier connection would otherwise show up in the reader).
            self.filters.retire({expiry_ms}, now_ms=int(time.time() * 1000))

            # Select strikes around spot
            strikes = [
                s for s in sorted(chain.strikes)
                if abs(s - spot) <= self.strike_radius and s % 5 == 0
            ]
            if not strikes:
                strikes = sorted(chain.strikes, key=lambda s: abs(s - spot))[:20]

            # Build OTM Contracts (Puts below spot, Calls above spot)
            contracts = []
            for s in strikes:
                right = "P" if s < spot else "C"
                contracts.append(
                    Option(
                        "SPX",
                        expiry_str,
                        s,
                        right,
                        chain.exchange,
                        tradingClass=chain.tradingClass,
                    )
                )

            await ib.qualifyContractsAsync(*contracts)
            tickers = [ib.reqMktData(c, genericTickList="100,101,106") for c in contracts]

            self.snapshot = {
                **self.snapshot,
                "status": (
                    f"{self._status_prefix()}Subscribed to {len(contracts)} "
                    f"SPX 0DTE contracts ({expiry_str})"
                ),
                "connected": True,
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
                    _finite_positive(ticker_spx.marketPrice())
                    or _finite_positive(ticker_spx.close)
                    or spot
                )

                time_to_close_ms = expiry_ms - now_ms
                if time_to_close_ms <= MIN_SECONDS_TO_EXPIRY * 1000:
                    self.filters.retire(set(), now_ms=now_ms)
                    self.snapshot = {
                        **self.snapshot,
                        "status": (
                            f"SPX {expiry_str} session closed (4:00 PM ET) — "
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

                latest_quotes = {}

                for c, t in zip(contracts, tickers):
                    if t.bid and t.ask and t.bid > 0 and t.ask >= t.bid:
                        mid = (t.bid + t.ask) / 2.0
                        opt_type = OptionType.Put if c.right == "P" else OptionType.Call
                        # Invert bid/ask/mid with Black-76. A quote whose bid or ask
                        # cannot be inverted is skipped, never filled with a
                        # made-up spread (it would set the filter's precision).
                        bid_iv = _implied_vol(opt_type, c.strike, spot, t_years, t.bid)
                        ask_iv = _implied_vol(opt_type, c.strike, spot, t_years, t.ask)
                        if bid_iv is None or ask_iv is None:
                            continue
                        mid_iv = _implied_vol(opt_type, c.strike, spot, t_years, mid)
                        if mid_iv is None:
                            mid_iv = (bid_iv + ask_iv) / 2.0
                        if ask_iv < bid_iv:
                            bid_iv, ask_iv = ask_iv, bid_iv
                        mid_iv = max(bid_iv, min(ask_iv, mid_iv))

                        name = f"SPX_{expiry_str}_{int(c.strike)}_{c.right}"
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
                            # Spot stands in for the forward (discount = 1); the
                            # carry error is a few index points at most over
                            # <= 6.5h and is not corrected here.
                            forward=spot,
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
                    "status": (
                        f"{self._status_prefix()}Streaming 0DTE quotes "
                        f"({len(latest_quotes)} active strikes{rejects})"
                    ),
                    "connected": True,
                    "spot_price": spot,
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

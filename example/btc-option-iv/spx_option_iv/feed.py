"""SPX 0DTE live feed controller with IBKR socket and Bayesian Kalman filter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time
from zoneinfo import ZoneInfo

# Initialize event loop for Python 3.14 before importing ib_async
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_async import IB, Index, Option
from itofin import ItofinError
from itofin.instruments import OptionType
from itofin.pricingengines import black_formula_implied_volatility

from btc_option_iv.kalman import FilterConfig, FilterManager


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
    bid_premium_btc: float  # named bid_premium_btc to match FilterManager fingerprint
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

    async def _run(self):
        ib = IB()
        try:
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
                raise RuntimeError(f"Unable to connect to IB Gateway (clientIds in use)")
            ib.reqMarketDataType(3)  # 3 = Delayed (free fallback), 1 = Real-time if subscribed

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
            # Prefer SPXW (weekly/daily) contracts
            chain = next(
                (c for c in chains if c.tradingClass == "SPXW" and c.exchange == "SMART"),
                chains[0] if chains else None,
            )
            if not chain:
                raise RuntimeError("No SPXW option chains found from IBKR")

            expirations = sorted(chain.expirations)
            expiry_str = expirations[0]  # Nearest 0DTE expiration
            expiry_ms = compute_0dte_expiry_ms(expiry_str)

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
                    Option("SPX", expiry_str, s, right, "SMART", tradingClass="SPXW")
                )

            await ib.qualifyContractsAsync(*contracts)
            tickers = [ib.reqMktData(c, genericTickList="100,101,106") for c in contracts]

            self.snapshot = {
                **self.snapshot,
                "status": f"Subscribed to {len(contracts)} SPX 0DTE contracts ({expiry_str})",
                "connected": True,
                "expiry_str": expiry_str,
                "expiry_timestamp_ms": expiry_ms,
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

                time_to_close_ms = max(1000, expiry_ms - now_ms)
                hours_left = time_to_close_ms / (3600 * 1000)
                t_years = max(1e-4, time_to_close_ms / (365.25 * 86_400_000))

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

                        try:
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
                                forward=spot,
                                expiry=t_years,
                                bid_premium_btc=t.bid,
                                ask_premium_btc=t.ask,
                                bid_usd=t.bid,
                                ask_usd=t.ask,
                                right=c.right,
                            )
                            latest_quotes[name] = row
                            self.filters.ingest(row, inst, now_ms=now_ms, now=now)
                        except Exception:
                            continue

                # Run Kalman prediction tick across active filters
                self.filters.tick(now_ms=now_ms, now=now)

                self.snapshot = {
                    "status": f"Streaming live 0DTE quotes ({len(latest_quotes)} active strikes)",
                    "connected": True,
                    "spot_price": spot,
                    "expiry_str": expiry_str,
                    "expiry_timestamp_ms": expiry_ms,
                    "time_to_close_hours": hours_left,
                    "latest_quotes": latest_quotes,
                    "kalman_views": self.filters.snapshots(now=now),
                    "filter_config": self.filters.config,
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
            if ib.isConnected():
                ib.disconnect()

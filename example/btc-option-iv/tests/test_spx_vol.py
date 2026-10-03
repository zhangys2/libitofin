"""Unit tests for SPX 0DTE feed controller, views, and Kalman smile calibration."""

from __future__ import annotations

import math
import time
import numpy as np
import pandas as pd
import pytest

from btc_option_iv.kalman import FilterConfig, FilterManager
import asyncio
from types import SimpleNamespace

from itofin.instruments import OptionType

import spx_option_iv.feed as feed
from spx_option_iv.feed import (
    SpxFeedController,
    SpxInstrumentMeta,
    SpxIvRow,
    _finite_positive,
    _implied_vol,
    parity_forward,
    compute_0dte_expiry_ms,
    select_live_expiry,
    select_spxw_chain,
)
from btc_option_iv.reference import butterfly_markdown, butterfly_report, reference_vol
from spx_option_iv.views import make_spx_smile_figure, make_spx_smile_reader


def test_compute_0dte_expiry_ms():
    expiry_ms = compute_0dte_expiry_ms("20261002")
    # 2026-10-02 16:00 EDT == 20:00 UTC
    assert expiry_ms == 1790971200 * 1000


def test_spx_smile_reader_empty_snapshot():
    analysis, error = make_spx_smile_reader({})
    assert analysis is None
    assert "Waiting" in error


def test_spx_synthetic_quotes_ingest_and_reader():
    manager = FilterManager()
    now_ms = int(time.time() * 1000)
    now = time.monotonic()
    expiry_ms = now_ms + 4 * 3600 * 1000
    t_years = 4.0 / (365.25 * 24)
    fwd = 7740.0

    strikes = [7680, 7700, 7720, 7740, 7760, 7780, 7800]
    for s in strikes:
        mid_iv = 0.18 + 0.05 * ((s - fwd) / fwd) ** 2 - 0.10 * ((s - fwd) / fwd)
        bid_iv = mid_iv - 0.005
        ask_iv = mid_iv + 0.005
        name = f"SPX_20261002_{s}_C"
        inst = SpxInstrumentMeta(name, expiry_ms, "c", float(s))
        row = SpxIvRow(
            instrument_name=name,
            timestamp_ms=now_ms,
            strike=float(s),
            mid_iv=mid_iv,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            forward=fwd,
            expiry=t_years,
            bid_premium_btc=5.0,
            ask_premium_btc=5.2,
            bid_usd=5.0,
            ask_usd=5.2,
            right="Call",
        )
        manager.ingest(row, inst, now_ms=now_ms, now=now)

    views = manager.snapshots(now=now)
    assert len(views) == 1
    view = list(views.values())[0]
    assert view.knot_ivs is not None
    assert len(view.knot_ivs) == 9
    assert view.reference_section is not None

    snapshot = {
        "status": "Streaming live 0DTE quotes",
        "connected": True,
        "spot_price": fwd,
        "expiry_str": "20261002",
        "expiry_timestamp_ms": expiry_ms,
        "time_to_close_hours": 4.0,
        "latest_quotes": {},
        "kalman_views": views,
        "feed_running": True,
    }

    analysis, error = make_spx_smile_reader(snapshot)
    assert error is None
    assert analysis is not None
    assert analysis["spot_price"] == fwd
    assert analysis["atm_vol"] == pytest.approx(0.18, abs=0.01)
    assert len(analysis["knots"]) == 9
    assert len(analysis["observations"]) == len(strikes)

    obs_df = analysis["observations"]
    assert "strike_usd" in obs_df.columns
    assert "observed_mid_iv" in obs_df.columns
    assert "fitted_iv" in obs_df.columns
    assert "filtered_iv" in obs_df.columns
    assert "fitted_minus_observed" in obs_df.columns

    fig = make_spx_smile_figure(analysis)
    assert fig is not None
    trace_names = [t.name for t in fig.data]
    assert any("Total Variance Smile" in n for n in trace_names)
    assert any("Kalman Filtered Smile" in n for n in trace_names)
    assert any("Observed Mid IV" in n for n in trace_names)
    assert any("Kalman State Knots" in n for n in trace_names)


def test_make_spx_smile_figure_error():
    fig = make_spx_smile_figure(None, error="Custom error message")
    assert fig is not None
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "Custom error message"


def test_finite_positive_rejects_nan_and_nonpositive():
    nan = float("nan")
    assert _finite_positive(nan) is None
    assert _finite_positive(None) is None
    assert _finite_positive(0.0) is None
    assert _finite_positive(-1.0) is None
    assert _finite_positive(float("inf")) is None
    assert _finite_positive(7700) == 7700.0
    # The original bug: NaN is truthy, so `nan or fallback` kept NaN.
    assert (nan or 5.0) is nan
    assert (_finite_positive(nan) or 5.0) == 5.0


def test_implied_vol_returns_none_when_not_invertible():
    # A price below intrinsic value cannot be inverted.
    assert _implied_vol(OptionType.Call, 7000.0, 7740.0, 0.001, 1.0) is None
    iv = _implied_vol(OptionType.Call, 7740.0, 7740.0, 0.001, 10.0)
    assert iv is not None and iv > 0


def _black76(opt, strike, forward, t, vol):
    n = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    sd = vol * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if opt == "C":
        return forward * n(d1) - strike * n(d2)
    return strike * n(-d2) - forward * n(-d1)


class _FakeTicker:
    def __init__(self, market_price=float("nan"), close=float("nan"), bid=None, ask=None):
        self._market_price = market_price
        self.close = close
        self.bid, self.ask = bid, ask
        self.modelGreeks = None

    def marketPrice(self):
        return self._market_price


class _FakeIB:
    """Minimal ib_async stand-in; option tickers are built by ``option_ticker``."""

    last = None

    def __init__(self, spot_ticker, option_ticker=None, chains=()):
        self.spot_ticker = spot_ticker
        self.option_ticker = option_ticker
        self.chains = list(chains)
        self.market_data_type = None
        self.option_contracts = []
        _FakeIB.last = self

    async def connectAsync(self, *args, **kwargs):
        pass

    def reqMarketDataType(self, kind):
        self.market_data_type = kind

    async def qualifyContractsAsync(self, *contracts):
        pass

    async def reqSecDefOptParamsAsync(self, *args):
        return self.chains

    def reqMktData(self, contract, *args, **kwargs):
        if isinstance(contract, _FakeIndex):
            return self.spot_ticker
        self.option_contracts.append(contract)
        return self.option_ticker(contract)

    def isConnected(self):
        return False


class _FakeIndex:
    conId = 1

    def __init__(self, *args):
        pass


class _FakeOption:
    def __init__(self, symbol, expiry, strike, right, exchange, tradingClass=None):
        self.strike, self.right = float(strike), right
        self.exchange, self.tradingClass = exchange, tradingClass


EXPIRY_STR = "20261002"
CLOSE_MS = compute_0dte_expiry_ms(EXPIRY_STR)


def _spxw_chain():
    return SimpleNamespace(
        tradingClass="SPXW",
        exchange="SMART",
        expirations=[EXPIRY_STR],
        strikes=[float(k) for k in range(7690, 7791, 5)],
    )


def _run_feed(monkeypatch, fake_ib, *, now_ms, loops, market_data_type=1):
    """Run the feed with a fake IB at a frozen wall clock for ``loops`` polls."""
    sleeps = {"n": 0}

    async def fake_sleep(_):
        sleeps["n"] += 1
        # 1 spot-wait sleep, then one per loop; cancel after ``loops`` polls.
        if sleeps["n"] > 1 + loops:
            raise asyncio.CancelledError

    monkeypatch.setattr(feed.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(feed.time, "time", lambda: now_ms / 1000.0)
    monkeypatch.setattr(
        feed, "_load_ib", lambda: (lambda: fake_ib, _FakeIndex, _FakeOption)
    )
    published = []
    controller = SpxFeedController(published.append)
    controller.set_market_data_type(market_data_type)
    asyncio.run(controller._run())
    return controller, published


def _smart_option_ticker(now_ms, forward=7740.0, vol=0.2):
    """Tickers priced off ``forward`` (which may differ from the index spot)."""
    t = (CLOSE_MS - now_ms) / feed.YEAR_MS

    def make(contract):
        px = _black76(contract.right, contract.strike, forward, t, vol)
        return _FakeTicker(bid=px * 0.99, ask=px * 1.01)

    return make


def test_feed_errors_instead_of_using_nan_spot(monkeypatch):
    fake = _FakeIB(_FakeTicker())
    # The spot wait polls 10 times before giving up.
    controller, _ = _run_feed(monkeypatch, fake, now_ms=CLOSE_MS - 3_600_000, loops=20)
    assert "No SPX spot price" in controller.snapshot["status"]
    assert controller.snapshot["connected"] is False
    assert controller.snapshot["feed_running"] is False


def test_feed_requests_live_market_data_by_default_and_flags_delayed(monkeypatch):
    now_ms = CLOSE_MS - 3_600_000
    fake = _FakeIB(
        _FakeTicker(market_price=7740.0),
        _smart_option_ticker(now_ms),
        [_spxw_chain()],
    )
    controller, _ = _run_feed(monkeypatch, fake, now_ms=now_ms, loops=1)
    assert fake.market_data_type == 1
    assert "not live" not in controller.snapshot["status"]

    fake = _FakeIB(
        _FakeTicker(market_price=7740.0),
        _smart_option_ticker(now_ms),
        [_spxw_chain()],
    )
    controller, _ = _run_feed(
        monkeypatch, fake, now_ms=now_ms, loops=1, market_data_type=3
    )
    assert fake.market_data_type == 3
    assert "Delayed" in controller.snapshot["status"]
    assert "not live" in controller.snapshot["status"]


def test_feed_inverts_with_real_remaining_time_and_builds_from_chain(monkeypatch):
    # 10 minutes left: the old 1e-4y (~53 min) floor would bias every IV low.
    now_ms = CLOSE_MS - 10 * 60_000
    fake = _FakeIB(
        _FakeTicker(market_price=7740.0),
        _smart_option_ticker(now_ms, vol=0.2),
        [_spxw_chain()],
    )
    controller, _ = _run_feed(monkeypatch, fake, now_ms=now_ms, loops=1)
    quotes = controller.snapshot["latest_quotes"]
    assert quotes
    expected_t = 10 * 60_000 / feed.YEAR_MS
    for row in quotes.values():
        assert row.expiry == pytest.approx(expected_t)
    # Near-ATM strikes are well-conditioned at 10 minutes (sd ~ 7 pts); the old
    # floor would have returned ~0.09 here.
    near_atm = [r for r in quotes.values() if abs(r.strike - 7740.0) <= 10]
    assert near_atm
    for row in near_atm:
        assert row.mid_iv == pytest.approx(0.2, abs=5e-3)
    assert all(c.tradingClass == "SPXW" and c.exchange == "SMART" for c in fake.option_contracts)
    assert controller.snapshot["time_to_close_hours"] == pytest.approx(10 / 60)
    # Only the subscribed expiry owns a filter.
    assert set(controller.snapshot["kalman_views"]) <= {CLOSE_MS}


def test_feed_stops_with_closed_status_at_the_close(monkeypatch):
    now_ms = CLOSE_MS - 10_000  # inside the 30s cutoff
    fake = _FakeIB(
        _FakeTicker(market_price=7740.0),
        _smart_option_ticker(CLOSE_MS - 3_600_000),
        [_spxw_chain()],
    )
    controller, published = _run_feed(monkeypatch, fake, now_ms=now_ms, loops=3)
    assert "session closed" in controller.snapshot["status"]
    assert controller.snapshot["feed_running"] is False
    assert controller.snapshot["latest_quotes"] == {}
    assert controller.snapshot["kalman_views"] == {}


def test_feed_skips_quotes_whose_bid_or_ask_iv_cannot_invert(monkeypatch):
    now_ms = CLOSE_MS - 3_600_000
    good = _smart_option_ticker(now_ms)

    def make(contract):
        if contract.strike == 7740.0:
            # A call priced above its forward cannot be inverted.
            return _FakeTicker(bid=9000.0, ask=9100.0)
        return good(contract)

    fake = _FakeIB(_FakeTicker(market_price=7740.0), make, [_spxw_chain()])
    controller, _ = _run_feed(monkeypatch, fake, now_ms=now_ms, loops=1)
    names = list(controller.snapshot["latest_quotes"])
    assert names
    assert not any("_7740_" in n for n in names)


def test_feed_refuses_non_spxw_chain(monkeypatch):
    monthly = SimpleNamespace(
        tradingClass="SPX", exchange="SMART", expirations=[EXPIRY_STR], strikes=[7740.0]
    )
    fake = _FakeIB(_FakeTicker(market_price=7740.0), None, [monthly])
    controller, _ = _run_feed(monkeypatch, fake, now_ms=CLOSE_MS - 3_600_000, loops=1)
    assert "No SPXW option chains" in controller.snapshot["status"]
    assert fake.option_contracts == []


def test_reader_shows_only_the_subscribed_expiry():
    manager = FilterManager()
    now_ms = int(time.time() * 1000)
    now = time.monotonic()
    fwd = 7740.0
    expiries = {}
    for days, base_vol in ((0, 0.18), (1, 0.30)):
        expiry_ms = now_ms + 4 * 3600 * 1000 + days * 86_400_000
        expiries[days] = expiry_ms
        t_years = (expiry_ms - now_ms) / feed.YEAR_MS
        for s in [7680, 7700, 7720, 7740, 7760, 7780, 7800]:
            mid_iv = base_vol + 0.05 * ((s - fwd) / fwd) ** 2
            name = f"SPX_{days}_{s}_C"
            inst = SpxInstrumentMeta(name, expiry_ms, "c", float(s))
            row = SpxIvRow(
                instrument_name=name, timestamp_ms=now_ms, strike=float(s),
                mid_iv=mid_iv, bid_iv=mid_iv - 0.005, ask_iv=mid_iv + 0.005,
                forward=fwd, expiry=t_years, bid_premium_btc=5.0, ask_premium_btc=5.2,
                bid_usd=5.0, ask_usd=5.2, right="C",
            )
            manager.ingest(row, inst, now_ms=now_ms, now=now)
    views = manager.snapshots(now=now)
    assert set(views) == set(expiries.values())
    base = {
        "status": "Streaming", "spot_price": fwd, "expiry_str": "x",
        "time_to_close_hours": 4.0, "latest_quotes": {}, "kalman_views": views,
    }
    # The later expiry is inserted second; the reader must follow the snapshot's expiry.
    near, _ = make_spx_smile_reader({**base, "expiry_timestamp_ms": expiries[0]})
    far, _ = make_spx_smile_reader({**base, "expiry_timestamp_ms": expiries[1]})
    assert near["atm_vol"] == pytest.approx(0.18, abs=0.02)
    assert far["atm_vol"] == pytest.approx(0.30, abs=0.02)
    missing, error = make_spx_smile_reader({**base, "expiry_timestamp_ms": 1})
    assert missing is None and error == "Streaming"


def _parity_pairs(forward, strikes, t=1.0 / 8766, vol=0.2, rel_spread=0.01):
    pairs = []
    for k in strikes:
        c, p = _black76("C", k, forward, t, vol), _black76("P", k, forward, t, vol)
        pairs.append((k, c * (1 - rel_spread), c * (1 + rel_spread),
                      p * (1 - rel_spread), p * (1 + rel_spread)))
    return pairs


def test_parity_forward_recovers_the_forward_and_is_robust():
    strikes = [7720.0, 7730.0, 7740.0, 7750.0, 7760.0]
    result = parity_forward(_parity_pairs(7745.0, strikes), spot=7740.0)
    assert result is not None
    forward, used = result
    assert used == 5
    assert forward == pytest.approx(7745.0, abs=1e-6)
    # One stale pair (wide spread, wrong prices) is outvoted by the median.
    pairs = _parity_pairs(7745.0, strikes)
    k, cb, ca, pb, pa = pairs[2]
    pairs[2] = (k, cb + 8, ca + 12, pb, pa)
    forward, _ = parity_forward(pairs, spot=7740.0)
    assert forward == pytest.approx(7745.0, abs=1e-6)


def test_parity_forward_returns_none_when_unusable():
    spot = 7740.0
    assert parity_forward([], spot) is None
    assert parity_forward(_parity_pairs(7745.0, [7740.0]), spot) is None  # < 2 pairs
    # Implausible forward (>2% from spot) is rejected.
    assert parity_forward(_parity_pairs(8000.0, [7980.0, 7990.0, 8000.0]), spot) is None
    # Premiums that are not near-ATM option prices (bad quotes) are ignored.
    bad = [(7740.0, 9000.0, 9100.0, 9000.0, 9100.0), (7750.0, 9000.0, 9100.0, 9000.0, 9100.0)]
    assert parity_forward(bad, spot) is None


def test_feed_uses_parity_forward_for_inversion(monkeypatch):
    now_ms = CLOSE_MS - 3_600_000
    # Index spot 7740 but the market is priced off a 7746 forward.
    fake = _FakeIB(
        _FakeTicker(market_price=7740.0),
        _smart_option_ticker(now_ms, forward=7746.0, vol=0.2),
        [_spxw_chain()],
    )
    controller, _ = _run_feed(monkeypatch, fake, now_ms=now_ms, loops=1)
    snap = controller.snapshot
    assert snap["forward"] == pytest.approx(7746.0, abs=0.05)
    assert snap["spot_price"] == 7740.0
    assert "parity" in snap["forward_source"]
    quotes = snap["latest_quotes"]
    assert {"C", "P"} <= {r.right for r in quotes.values()}
    for row in quotes.values():
        assert row.forward == pytest.approx(7746.0, abs=0.05)
        if abs(row.strike - 7746.0) <= 30:
            assert row.mid_iv == pytest.approx(0.2, abs=5e-3)
    # Both rights are subscribed only at the strikes nearest the money.
    both = {c.strike for c in fake.option_contracts if c.right == "C"} & {
        c.strike for c in fake.option_contracts if c.right == "P"
    }
    assert len(both) == feed.PARITY_STRIKES


def test_feed_falls_back_to_spot_when_parity_is_unavailable(monkeypatch):
    now_ms = CLOSE_MS - 3_600_000
    good = _smart_option_ticker(now_ms)

    def calls_only(contract):
        # No two-sided puts anywhere: no parity pair can form.
        return good(contract) if contract.right == "C" else _FakeTicker()

    fake = _FakeIB(_FakeTicker(market_price=7740.0), calls_only, [_spxw_chain()])
    controller, _ = _run_feed(monkeypatch, fake, now_ms=now_ms, loops=1)
    snap = controller.snapshot
    assert snap["forward"] == 7740.0
    assert "spot" in snap["forward_source"]

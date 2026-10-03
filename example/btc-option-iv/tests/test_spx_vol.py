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
        inst = SpxInstrumentMeta(name, expiry_ms, "call", float(s))
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


class _FakeTicker:
    def __init__(self, market_price=float("nan"), close=float("nan")):
        self._market_price = market_price
        self.close = close
        self.bid = self.ask = None
        self.modelGreeks = None

    def marketPrice(self):
        return self._market_price


class _FakeIB:
    def __init__(self, ticker):
        self.ticker = ticker

    async def connectAsync(self, *args, **kwargs):
        pass

    def reqMarketDataType(self, _):
        pass

    async def qualifyContractsAsync(self, *contracts):
        pass

    def reqMktData(self, *args, **kwargs):
        return self.ticker

    def isConnected(self):
        return False


def test_feed_errors_instead_of_using_nan_spot(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(feed.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(feed, "IB", lambda: _FakeIB(_FakeTicker()))
    published = []
    controller = SpxFeedController(published.append)
    asyncio.run(controller._run())
    assert "No SPX spot price" in controller.snapshot["status"]
    assert controller.snapshot["connected"] is False
    assert controller.snapshot["feed_running"] is False


def _chain(trading_class, exchange):
    return SimpleNamespace(tradingClass=trading_class, exchange=exchange)


def test_select_spxw_chain_never_falls_back_to_monthly_spx():
    monthly, smart, other = (
        _chain("SPX", "SMART"),
        _chain("SPXW", "SMART"),
        _chain("SPXW", "CBOE"),
    )
    assert select_spxw_chain([monthly, other, smart]) is smart
    assert select_spxw_chain([monthly, other]) is other
    assert select_spxw_chain([monthly]) is None
    assert select_spxw_chain([]) is None


def test_select_live_expiry_skips_expired_dates():
    close = compute_0dte_expiry_ms("20261002")
    assert select_live_expiry(["20261005", "20261002", "20261001"], close - 1) == (
        "20261002",
        close,
    )
    # After today's close the expired date is skipped.
    assert select_live_expiry(["20261005", "20261002"], close + 1) == (
        "20261005",
        compute_0dte_expiry_ms("20261005"),
    )
    assert select_live_expiry(["20261002"], close + 1) is None


class _RaisingSection:
    @property
    def butterfly_report(self):
        raise ValueError("boom")

    def volatility(self, strike):
        raise ValueError("boom")


def test_reference_helpers_degrade_to_none_on_errors():
    missing = SimpleNamespace(reference_section=None)
    broken = SimpleNamespace(reference_section=_RaisingSection())
    for view in (missing, broken):
        assert reference_vol(view, 7700.0) is None
        assert butterfly_report(view) is None
    assert butterfly_markdown(None) == ""
    report = SimpleNamespace(
        has_arbitrage=False, min_density=0.25, final_smoothing=0.01, ramp_iterations=0
    )
    assert "Arbitrage-free" in butterfly_markdown(report)
    report.has_arbitrage = True
    assert "Arbitrage detected" in butterfly_markdown(report, label="X").split(":")[1]

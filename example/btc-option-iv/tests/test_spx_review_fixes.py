from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import spx_option_iv.feed as feed
from spx_option_iv.feed import (
    _select_contract_expiry,
    SpxFeedController,
    SpxInstrumentMeta,
    SpxIvRow,
    _observe_quote,
    _subscribe_quote_updates,
    compute_0dte_expiry_ms,
    parity_forward,
    select_strikes,
)


def test_expiry_close_includes_recurring_half_days_and_not_unlisted_dates():
    ny = timezone.utc
    assert datetime.fromtimestamp(
        compute_0dte_expiry_ms("20261127") / 1000, ny
    ).hour == 18  # 1 PM EST
    assert datetime.fromtimestamp(
        compute_0dte_expiry_ms("20271126") / 1000, ny
    ).hour == 18  # 1 PM EST
    assert datetime.fromtimestamp(
        compute_0dte_expiry_ms("20261224") / 1000, ny
    ).hour == 18
    # Cboe's 2026 schedule has no July 2 early close.
    assert datetime.fromtimestamp(
        compute_0dte_expiry_ms("20260702") / 1000, ny
    ).hour == 20
    assert datetime.fromtimestamp(
        compute_0dte_expiry_ms("20280703") / 1000, ny
    ).hour == 17  # 1 PM EDT


def test_expiry_uses_authoritative_hours_for_exceptional_early_close():
    expiry = "20261116"  # not a recurring half-day in the calendar helper
    close_ms = int(
        datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc).timestamp() * 1000
    )
    now_ms = int(
        datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc).timestamp() * 1000
    )

    class IB:
        async def qualifyContractsAsync(self, contract):
            pass

        async def reqContractDetailsAsync(self, contract):
            return [
                SimpleNamespace(
                    timeZoneId="US/Central",
                    lastTradeTime="15:00:00",
                    liquidHours=f"{expiry}:0830-{expiry}:1200",
                )
            ]

    chain = SimpleNamespace(
        expirations=[expiry], strikes=[100.0], exchange="SMART", tradingClass="SPXW"
    )
    selected = asyncio.run(
        _select_contract_expiry(
            IB(), lambda *args, **kwargs: SimpleNamespace(), "SPX", chain, 100.0, now_ms
        )
    )
    assert selected == (expiry, close_ms)


def test_expiry_fails_closed_when_contract_hours_are_missing():
    class IB:
        async def qualifyContractsAsync(self, contract):
            pass

        async def reqContractDetailsAsync(self, contract):
            return [SimpleNamespace(timeZoneId="US/Central", liquidHours="")]

    chain = SimpleNamespace(
        expirations=["20261116"],
        strikes=[100.0],
        exchange="SMART",
        tradingClass="SPXW",
    )
    with pytest.raises(RuntimeError, match="session hours are incomplete"):
        asyncio.run(
            _select_contract_expiry(
                IB(),
                lambda *args, **kwargs: SimpleNamespace(),
                "SPX",
                chain,
                100.0,
                int(datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc).timestamp() * 1000),
            )
        )


def test_quote_observation_expires_withdraws_and_uses_bid_ask_ticks_only():
    now_ms = 1_800_000_000_000
    stamp = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    state = {}
    ticker = SimpleNamespace(
        ticks=[
            SimpleNamespace(tickType=1, time=stamp, price=10.0),
            SimpleNamespace(tickType=2, time=stamp, price=11.0),
            # Trade ticks are not bid/ask quote timestamps.
            SimpleNamespace(tickType=68, time=stamp, price=10.5),
        ]
    )
    assert _observe_quote(ticker, state, now_ms) == (10.0, 11.0)
    assert _observe_quote(SimpleNamespace(ticks=[]), state, now_ms + 10_001) is None

    withdrawal = SimpleNamespace(
        ticks=[
            SimpleNamespace(
                tickType=1,
                time=datetime.fromtimestamp((now_ms + 1000) / 1000, timezone.utc),
                price=0,
            )
        ]
    )
    assert _observe_quote(withdrawal, state, now_ms + 1000) is None


def test_delayed_bid_ask_ticks_require_their_own_observation_times():
    now_ms = 1_800_000_000_000
    stamp = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    state = {}
    delayed = SimpleNamespace(
        ticks=[
            SimpleNamespace(tickType=66, time=stamp, price=10.0),
            SimpleNamespace(tickType=67, time=stamp, price=11.0),
        ]
    )
    assert _observe_quote(delayed, state, now_ms) == (10.0, 11.0)
    assert state["bid_time"] == now_ms
    assert state["bid_source_time"] is None
    assert state["ask_source_time"] is None
    # Delayed trade timestamps alone cannot manufacture a quote observation.
    no_quote_time = _observe_quote(
        SimpleNamespace(ticks=[SimpleNamespace(tickType=68, time=stamp, price=10.5)]),
        {},
        now_ms,
    )
    assert no_quote_time is None


def test_tick_callback_retains_bid_and_ask_from_separate_cleared_packets():
    class Event:
        def __init__(self):
            self.callbacks = []

        def __iadd__(self, callback):
            self.callbacks.append(callback)
            return self

        def emit(self, ticker):
            for callback in self.callbacks:
                callback(ticker)

    now_ms = 1_800_000_000_000
    stamp = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    ticker = SimpleNamespace(ticks=[], updateEvent=Event())
    observation = {}
    _subscribe_quote_updates(ticker, observation)

    ticker.ticks = [SimpleNamespace(tickType=1, time=stamp, price=10.0)]
    ticker.updateEvent.emit(ticker)
    ticker.ticks.clear()  # ib_async clears this list at the next packet
    assert _observe_quote(ticker, observation, now_ms) is None

    ticker.ticks = [SimpleNamespace(tickType=2, time=stamp, price=11.0)]
    ticker.updateEvent.emit(ticker)
    ticker.ticks.clear()
    assert _observe_quote(ticker, observation, now_ms) == (10.0, 11.0)


def test_parity_rejects_diagnostic_invalid_even_when_forward_is_plausible(monkeypatch):
    monkeypatch.setattr(
        feed,
        "implied_forward",
        lambda **kwargs: SimpleNamespace(forward=100.0, pairs_used=2, is_valid=False),
    )
    pairs = [(100.0, 1.0, 1.1, 1.0, 1.1), (101.0, 1.0, 1.1, 1.0, 1.1)]
    assert parity_forward(pairs, 100.0) is None


def test_symbol_switch_clears_filter_quote_and_version_caches():
    controller = SpxFeedController(lambda _: None)
    now_ms = 1_800_000_000_000
    expiry_ms = now_ms + 60_000
    instrument = SpxInstrumentMeta("SPX_x_100_C", expiry_ms, "c", 100.0)
    row = SpxIvRow(
        instrument_name=instrument.instrument_name,
        timestamp_ms=now_ms,
        strike=100.0,
        mid_iv=0.2,
        bid_iv=0.19,
        ask_iv=0.21,
        forward=100.0,
        expiry=0.01,
        bid_premium_btc=1.0,
        ask_premium_btc=1.1,
        bid_usd=1.0,
        ask_usd=1.1,
        right="C",
    )
    controller.filters.ingest(row, instrument, now_ms=now_ms, now=1.0)
    assert controller.filters._filters
    assert controller.filters._quotes
    assert controller.filters._versions
    asyncio.run(controller.set_symbol("SPY"))
    assert controller.filters._filters == {}
    assert controller.filters._quotes == {}
    assert controller.filters._versions == {}
    assert controller.snapshot["latest_quotes"] == {}
    assert controller.snapshot["kalman_views"] == {}


def test_changing_strike_window_while_running_restarts_subscription():
    async def scenario():
        controller = SpxFeedController(lambda _: None)
        started = []
        never = asyncio.Event()

        async def run():
            started.append(controller.strike_radius)
            await never.wait()

        controller._run = run
        await controller.set_enabled(True)
        await asyncio.sleep(0)
        await controller.set_strike_radius(25.0)
        await asyncio.sleep(0)
        assert started == [80.0, 25.0]
        await controller.set_enabled(False)

    asyncio.run(scenario())


def test_strike_window_is_used_in_same_points_for_spy():
    selected = select_strikes(
        [float(x) for x in range(400, 601)], spot=500.0, radius=80.0, symbol="SPY"
    )
    assert min(selected) == 420.0
    assert max(selected) == 580.0


class _FeedFakeIB:
    def __init__(self, *, missing_hours=False):
        self.connected = False
        self.missing_hours = missing_hours

    async def connectAsync(self, *args, **kwargs):
        self.connected = True

    def reqMarketDataType(self, data_type):
        self.market_data_type = data_type

    async def qualifyContractsAsync(self, *contracts):
        pass

    async def reqSecDefOptParamsAsync(self, *args):
        return [
            SimpleNamespace(
                tradingClass="SPXW",
                exchange="SMART",
                expirations=["20261002"],
                strikes=[100.0],
            )
        ]

    async def reqContractDetailsAsync(self, contract):
        return [
            SimpleNamespace(
                timeZoneId="US/Eastern",
                lastTradeTime="16:00:00",
                liquidHours=(
                    ""
                    if self.missing_hours
                    else f"{contract.expiry}:0930-{contract.expiry}:1600"
                ),
            )
        ]

    def reqMktData(self, contract, *args, **kwargs):
        if isinstance(contract, _Underlying):
            return SimpleNamespace(marketPrice=lambda: 100.0, last=None, close=None)
        now_ms = compute_0dte_expiry_ms("20261002") - 3_600_000
        stamp = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
        tick_types = (66, 67) if self.market_data_type == 3 else ()
        ticks = [
            SimpleNamespace(tickType=kind, time=stamp, price=price)
            for kind, price in zip(tick_types, (1.0, 1.2))
        ]
        return SimpleNamespace(
            bid=1.0, ask=1.2, bidSize=1, askSize=1, ticks=ticks, modelGreeks=None
        )

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False


class _Underlying:
    conId = 1

    def __init__(self, *args):
        pass


class _Option:
    def __init__(self, *args, **kwargs):
        self.expiry = args[1]
        self.strike, self.right = float(args[2]), args[3]


def _run_minimal_feed(
    monkeypatch, *, data_type, disconnect=False, recorder=None, missing_hours=False
):
    now_ms = compute_0dte_expiry_ms("20261002") - 3_600_000
    fake_ib = _FeedFakeIB(missing_hours=missing_hours)
    sleeps = 0

    async def fake_sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if disconnect and sleeps == 2:
            fake_ib.connected = False
        if sleeps > 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(feed.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(feed.time, "time", lambda: now_ms / 1000)
    monkeypatch.setattr(
        feed, "_load_ib", lambda: (lambda: fake_ib, _Underlying, _Option, _Underlying)
    )
    recorded = []
    if recorder is not None:
        recorder.record_quote_async = lambda **kwargs: recorded.append(kwargs)
    controller = SpxFeedController(lambda _: None, recorder=recorder)
    controller.set_market_data_type(data_type)
    asyncio.run(controller._run())
    return controller, recorded


def test_delayed_feed_withholds_iv_without_timestamped_bid_ask(monkeypatch):
    controller, recorded = _run_minimal_feed(
        monkeypatch, data_type=3, recorder=SimpleNamespace()
    )
    assert "IV withheld" in controller.snapshot["status"]
    assert "delayed bid/ask quotes available" in controller.snapshot["status"]
    assert controller.snapshot["latest_quotes"] == {}
    assert controller.snapshot["available_quotes"]
    assert recorded == []


def test_feed_publishes_error_when_authoritative_expiry_hours_are_missing(monkeypatch):
    controller, _ = _run_minimal_feed(
        monkeypatch, data_type=1, missing_hours=True
    )
    assert "session hours are incomplete" in controller.snapshot["status"]
    assert controller.snapshot["feed_running"] is False


def test_feed_stops_and_publishes_error_when_ib_disconnects(monkeypatch):
    controller, _ = _run_minimal_feed(monkeypatch, data_type=1, disconnect=True)
    assert "IBKR disconnected" in controller.snapshot["status"]
    assert controller.snapshot["connected"] is False
    assert controller.snapshot["feed_running"] is False
    assert controller.snapshot["latest_quotes"] == {}

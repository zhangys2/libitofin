"""Offline feed/controller integration with synthetic source events."""

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import numpy as np
import scipy

from btc_option_iv import live
from btc_option_iv.market import IndexQuote, Instrument, OptionQuote
from test_kalman import CONTEXT, EXPIRY_MS, NOW_MS, quote


def test_feed_assimilates_events_before_publication_and_not_reference_updates(
    monkeypatch,
):
    published = []
    controller = live.LiveFeedController(published.append)
    instruments = [
        Instrument("left", "c", quote(-0.6)[0].strike, EXPIRY_MS, "BTC-FUTURE"),
        Instrument("right", "c", quote(0.6)[0].strike, EXPIRY_MS, "BTC-FUTURE"),
    ]
    monkeypatch.setattr(live, "fetch_active_btc_options", lambda: (instruments, []))
    wall = [NOW_MS]
    monkeypatch.setattr(live.time, "time", lambda: wall[0] / 1000)
    left = OptionQuote("left", NOW_MS, 0.295, 0.305, 1, 1)
    right = OptionQuote("right", NOW_MS, 0.295, 0.305, 1, 1)
    changed = replace(right, timestamp_ms=NOW_MS + 2, bid_price=0.296, ask_price=0.306)
    heartbeat = replace(changed, timestamp_ms=NOW_MS + 3, bid_amount=2)
    events = [
        left,
        right,
        right,
        IndexQuote(NOW_MS + 1, 100_000),
        changed,
        heartbeat,
        replace(changed, timestamp_ms=NOW_MS - 1, bid_price=0.1),
    ]
    observed = []

    async def stream(*args, **kwargs):
        for event in events:
            wall[0] = max(wall[0], event.timestamp_ms + 2)
            yield event
            views = controller.filters.snapshots(now=time.monotonic())
            if EXPIRY_MS in views:
                observed.append(views[EXPIRY_MS].updates)
        raise asyncio.CancelledError

    def evaluate(source, instrument, forwards, **kwargs):
        mid = (source.bid_price + source.ask_price) / 2
        revaluation = 0.001 if forwards.index is not None else 0
        return SimpleNamespace(
            instrument_name=source.instrument_name,
            timestamp_ms=source.timestamp_ms,
            strike=instrument.strike,
            mid_iv=mid + revaluation,
            bid_iv=mid + revaluation - 0.005,
            ask_iv=mid + revaluation + 0.005,
            bid_premium_btc=source.bid_price,
            ask_premium_btc=source.ask_price,
            forward=CONTEXT.forward,
            expiry=CONTEXT.exercise_time,
        )

    monkeypatch.setattr(live, "subscribe_btc_option_market_data", stream)
    monkeypatch.setattr(live, "evaluate_quote", evaluate)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller._run())
    assert observed == [0, 0, 0, 0, 1, 1, 1]
    # No selected-expiry UI was involved in creating or updating the filter.
    view = controller.filters.snapshots(now=time.monotonic())[EXPIRY_MS]
    assert len(view.knot_ivs) == 9 and view.updates == 1
    controller._republish()
    assert published[-1]["kalman_views"][EXPIRY_MS].updates == 1


def test_stopping_feed_preserves_states_and_cancels_stream(monkeypatch):
    controller = live.LiveFeedController(lambda snapshot: None)
    started = asyncio.Event()
    cancelled = []

    async def idle():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(controller, "_run", idle)

    async def exercise():
        manager = controller.filters
        await controller.set_enabled(True)
        await started.wait()
        first_task = controller.task
        await controller.set_enabled(True)
        assert controller.task is first_task
        await controller.set_enabled(False)
        assert controller.filters is manager
        assert controller.task is None
        assert not controller.snapshot["feed_running"]
        assert "frozen" in controller.snapshot["status"]

    asyncio.run(exercise())
    assert cancelled == [True]


def test_stale_snapshot_still_plots_last_observed_support():
    from btc_option_iv.kalman import FilterManager
    from btc_option_iv.views import make_smile_figure, make_smile_reader
    from test_kalman import warm

    manager = FilterManager()
    warm(manager)
    manager.tick(now_ms=NOW_MS + 11_000, now=11)
    snapshot = {
        "kalman_views": manager.snapshots(now=11),
        "feed_running": True,
        "published_ms": NOW_MS + 11_000,
    }
    analysis, error = make_smile_reader(snapshot, EXPIRY_MS, now_ms=NOW_MS + 11_000)
    assert error is None and analysis["filter_status"] == "stale"
    figure = make_smile_figure(analysis)
    assert figure.data[0].name == "Kalman filtered smile"
    assert min(figure.data[0].x) >= -1.4 - 1e-12
    assert max(figure.data[0].x) <= 1.4 + 1e-12


def test_configure_while_paused_does_not_reset_filter_on_long_pause(monkeypatch):
    from btc_option_iv.kalman import FilterConfig
    from test_kalman import warm

    published = []
    controller = live.LiveFeedController(published.append)
    warm(controller.filters)
    assert EXPIRY_MS in controller.filters.snapshots(now=1)

    controller.snapshot = {
        **controller.snapshot,
        "feed_running": False,
    }
    # Advance time by 120s during pause
    monkeypatch.setattr(live.time, "monotonic", lambda: 121.0)
    controller.configure(FilterConfig(process_iv_rate=0.20))

    # View must be preserved without triggering 60-second gap reset
    views = published[-1]["kalman_views"]
    assert EXPIRY_MS in views
    assert views[EXPIRY_MS].updates >= 1

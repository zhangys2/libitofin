"""Regression tests for asynchronous feed and recorder lifecycle failures."""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime
from pathlib import Path
import threading
from types import ModuleType
from typing import Any, Callable

import pytest

from spx_option_iv.feed import SpxFeedController
from spx_option_iv.recorder import ParquetL1Recorder


_ROOT = Path(__file__).resolve().parents[1]


def _level1_controller_type():
    source = (_ROOT / "level1_stream.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cell = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.ClassDef) and child.name == "Level1FeedController"
            for child in node.body
        )
    )
    cell.decorator_list = []
    cell.name = "_test_cell"
    namespace: dict[str, Any] = {
        "Any": Any,
        "Callable": Callable,
        "asyncio": asyncio,
        "datetime": datetime,
    }
    module = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
    exec(compile(module, str(_ROOT / "level1_stream.py"), "exec"), namespace)  # noqa: S102
    return namespace["_test_cell"](Any, Callable, asyncio, datetime)[0]


def test_spx_symbol_change_during_stop_does_not_orphan_replacement_task():
    async def scenario():
        controller = SpxFeedController(lambda _snapshot: None)
        old_started = asyncio.Event()
        old_cancelling = asyncio.Event()
        release_old = asyncio.Event()
        replacement_started = asyncio.Event()
        never = asyncio.Event()

        async def fake_run():
            if controller.symbol == "SPX":
                old_started.set()
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    old_cancelling.set()
                    await release_old.wait()
                    raise
            replacement_started.set()
            await never.wait()

        controller._run = fake_run
        await controller.set_enabled(True)
        await old_started.wait()

        stopping = asyncio.create_task(controller.set_enabled(False))
        await old_cancelling.wait()
        switching = asyncio.create_task(controller.set_symbol("SPY"))
        resuming = asyncio.create_task(controller.set_enabled(True))
        release_old.set()
        await asyncio.gather(stopping, switching, resuming)

        await asyncio.wait_for(replacement_started.wait(), timeout=1.0)
        assert controller.symbol == "SPY"
        assert controller.task is not None and not controller.task.done()
        assert controller.task.get_name() == "spy-0dte-live-feed"

        await controller.set_enabled(False)
        assert controller.task is None

    asyncio.run(scenario())


def test_level1_cancellation_during_qualification_disconnects_ib(monkeypatch):
    class FakeIB:
        def __init__(self):
            self.connected = False
            self.disconnected = False
            self.qualifying = asyncio.Event()

        async def connectAsync(self, *args, **kwargs):
            self.connected = True

        def reqMarketDataType(self, _kind):
            pass

        async def qualifyContractsAsync(self, *_contracts):
            self.qualifying.set()
            await asyncio.Event().wait()

        def isConnected(self):
            return self.connected

        def disconnect(self):
            self.connected = False
            self.disconnected = True

        def cancelMktData(self, _contract):
            pass

    fake_ib = FakeIB()

    class FakeContract:
        def __init__(self, *args):
            pass

    ib_async = ModuleType("ib_async")
    ib_async.IB = lambda: fake_ib
    ib_async.ContFuture = FakeContract
    ib_async.Crypto = FakeContract
    ib_async.Contract, ib_async.Stock = FakeContract, FakeContract
    monkeypatch.setitem(__import__("sys").modules, "ib_async", ib_async)
    controller_type = _level1_controller_type()
    controller = controller_type(lambda _state: None)

    async def scenario():
        await controller.set_enabled(True)
        await asyncio.wait_for(fake_ib.qualifying.wait(), timeout=1.0)
        assert controller.ib is fake_ib
        await controller.set_enabled(False)
        assert fake_ib.disconnected
        assert not fake_ib.connected
        assert controller.ib is None

    asyncio.run(scenario())


def test_recorder_async_record_does_not_block_event_loop(tmp_path, monkeypatch):
    recorder = ParquetL1Recorder(output_dir=tmp_path, batch_size_limit=1)
    loop = asyncio.new_event_loop()
    released = threading.Event()

    def blocking_record_quote(**_kwargs):
        loop.call_soon_threadsafe(released.set)
        if not released.wait(timeout=0.5):
            raise AssertionError("record_quote blocked the asyncio event loop")

    monkeypatch.setattr(recorder, "record_quote", blocking_record_quote)

    async def scenario():
        await asyncio.wait_for(
            recorder.record_quote_async(
                symbol="SPX",
                expiry="20261006",
                strike=100.0,
                right="C",
                bid=1.0,
                ask=1.1,
            ),
            timeout=1.0,
        )

    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()


def test_sync_recorder_context_exit_cancels_flush_task(tmp_path):
    async def scenario():
        recorder = ParquetL1Recorder(output_dir=tmp_path, flush_interval_sec=3600.0)
        with recorder:
            task = recorder._flush_task
            assert recorder._running
            assert task is not None and not task.done()
        assert not recorder._running
        assert recorder._flush_task is None
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

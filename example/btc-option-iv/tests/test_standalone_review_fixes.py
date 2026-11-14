"""Offline regressions for the IB historical and Level 1 tools."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from zoneinfo import ZoneInfo

from test_review_regressions import _level1_controller_type


_ROOT = Path(__file__).resolve().parents[1]


def test_partial_historical_download_preserves_existing_parquet(tmp_path):
    script = r'''
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
import pandas as pd
import download_historical_bars as downloader
from ib_async import BarData

class Event:
    def __iadd__(self, callback):
        return self

class FakeIB:
    def __init__(self):
        self.errorEvent = Event()
        self.calls = 0
    async def connectAsync(self, *args, **kwargs):
        pass
    async def qualifyContractsAsync(self, contract):
        return [contract]
    async def reqHistoricalDataAsync(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return [BarData(date=datetime(2025, 10, 1, tzinfo=timezone.utc),
                            open=10, high=11, low=9, close=10, volume=10)]
        return []
    def disconnect(self):
        pass

root = Path(sys.argv[1])
path = root / 'SPY.parquet'
original = pd.DataFrame({'date': pd.to_datetime(['2024-10-01', '2025-10-01'], utc=True),
                         'close': [9, 10]})
original.to_parquet(path)
downloader.IB = FakeIB
fetch = downloader.fetch_2y_5m_bars_for_symbol
async def fast_fetch(ib, contract):
    return await fetch(ib, contract, inter_chunk_pause=0)
downloader.fetch_2y_5m_bars_for_symbol = fast_fetch
sys.argv = ['download_historical_bars.py', '--symbols', 'SPY', '--output-dir', str(root)]
asyncio.run(downloader.main())
pd.testing.assert_frame_equal(pd.read_parquet(path), original)
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    assert "Successfully saved 0/1" in result.stdout


def test_level1_uses_front_dated_futures_and_reconnects(monkeypatch):
    today = datetime.now(ZoneInfo("America/New_York")).date()
    expired = (today - timedelta(days=1)).strftime("%Y%m%d")
    front = (today + timedelta(days=30)).strftime("%Y%m%d")
    later = (today + timedelta(days=60)).strftime("%Y%m%d")

    class Contract:
        def __init__(self, symbol, *args, exchange="", currency="", expiry=""):
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency
            self.secType = "FUT"
            self.lastTradeDateOrContractMonth = expiry
            self.localSymbol = symbol + expiry
            self.conId = int(expiry) if expiry else 0

    class FakeIB:
        instances = []
        created = None

        def __init__(self):
            self.connected = False
            self.requested = []
            self.ready = asyncio.Event()
            self.__class__.instances.append(self)
            self.__class__.created.set()

        async def connectAsync(self, host, port, **kwargs):
            self.connected = True
            self.port = port

        def reqMarketDataType(self, kind):
            self.market_data_type = kind

        async def qualifyContractsAsync(self, *contracts):
            return list(contracts)

        async def reqContractDetailsAsync(self, request):
            return [
                SimpleNamespace(
                    contract=Contract(request.symbol, expiry=expiry),
                    realExpirationDate=expiry,
                )
                for expiry in (later, expired, front)
            ]

        def reqMktData(self, contract):
            self.requested.append(contract)
            if {item.symbol for item in self.requested} >= {"CL", "GC"}:
                self.ready.set()
            return SimpleNamespace()

        def cancelMktData(self, contract):
            pass

        def isConnected(self):
            return self.connected

        def disconnect(self):
            self.connected = False

    ib_async = ModuleType("ib_async")
    ib_async.IB = FakeIB
    ib_async.Future = Contract
    ib_async.Crypto = Contract
    ib_async.Stock = Contract
    monkeypatch.setitem(sys.modules, "ib_async", ib_async)
    controller = _level1_controller_type()(lambda _snapshot: None, refresh_interval=3600)

    async def scenario():
        FakeIB.created = asyncio.Event()
        await controller.set_enabled(True)
        await asyncio.wait_for(FakeIB.created.wait(), timeout=2)
        first = FakeIB.instances[-1]
        await asyncio.wait_for(first.ready.wait(), timeout=2)
        futures = [contract for contract in first.requested if contract.symbol in {"CL", "GC"}]
        assert {(c.symbol, c.secType, c.lastTradeDateOrContractMonth) for c in futures} == {
            ("CL", "FUT", front),
            ("GC", "FUT", front),
        }
        controller.configure(port=4001, market_data_type=3, refresh_interval=3600)
        FakeIB.created = asyncio.Event()
        await controller.set_enabled(True)
        await asyncio.wait_for(FakeIB.created.wait(), timeout=2)
        second = FakeIB.instances[-1]
        await asyncio.wait_for(second.ready.wait(), timeout=2)
        assert not first.connected
        assert second.port == 4001
        assert second.market_data_type == 3
        await controller.set_enabled(False)
        assert not second.connected

    asyncio.run(scenario())

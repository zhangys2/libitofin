"""IB Gateway 1-Minute Bar Listener for Stocks and Crypto.

Connects to IB Gateway (or TWS) and streams live 1-minute OHLCV bars for
specified stocks and cryptocurrencies using ib-async.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import signal
import sys
from typing import Sequence

# Initialize loop policy before importing eventkit/ib_async to ensure
# compatibility across Python versions (including 3.12, 3.13, 3.14).
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_async import BarDataList, Contract, Crypto, IB, Stock, util  # noqa: E402


class BarListener:
    """Manages connections and 1-minute bar subscriptions for stocks & crypto."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 10,
        stocks: Sequence[str] = ("AAPL", "SPY"),
        cryptos: Sequence[str] = ("BTC", "ETH"),
        crypto_exchange: str = "PAXOS",
        currency: str = "USD",
        use_rth: bool = False,
        delayed: bool = False,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.stocks = [s.strip().upper() for s in stocks if s.strip()]
        self.cryptos = [c.strip().upper() for c in cryptos if c.strip()]
        self.crypto_exchange = crypto_exchange.upper()
        self.currency = currency.upper()
        self.use_rth = use_rth
        self.delayed = delayed

        self.ib = IB()
        self.active_subscriptions: dict[str, BarDataList] = {}
        self._running = False

    async def start(self) -> None:
        """Connect to IB Gateway and start listening to 1-minute bars."""
        print(f"Connecting to IB Gateway at {self.host}:{self.port} (clientId={self.client_id})...")
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=10, readonly=True)
        print(f"Connected to IB Gateway! Managed accounts: {self.ib.managedAccounts()}")

        if self.delayed:
            print("Enabling delayed market data (Type 3)...")
            self.ib.reqMarketDataType(3)

        # Register error handler to track API warnings/errors
        self.ib.errorEvent += self._on_error

        self._running = True

        # Build list of contracts to subscribe
        contracts_to_subscribe: list[tuple[str, Contract, str]] = []

        # 1. Prepare Stock contracts
        for sym in self.stocks:
            contract = Stock(sym, exchange="SMART", currency=self.currency)
            contracts_to_subscribe.append((f"STK:{sym}", contract, "TRADES"))

        # 2. Prepare Crypto contracts
        for sym in self.cryptos:
            contract = Crypto(sym, exchange=self.crypto_exchange, currency=self.currency)
            # Paxos crypto live updates require MIDPOINT (AGGTRADES does not support keepUpToDate=True)
            contracts_to_subscribe.append((f"CRYPTO:{sym}", contract, "MIDPOINT"))

        # Qualify and subscribe
        for label, contract, what_to_show in contracts_to_subscribe:
            await self._subscribe_bars(label, contract, what_to_show)

        if not self.active_subscriptions:
            print("[WARN] No active bar subscriptions. Check contract names or market data permissions.")
        else:
            print("\n" + "=" * 80)
            print(f"Streaming live 1-minute bars for {len(self.active_subscriptions)} contracts:")
            for k in self.active_subscriptions:
                print(f"  * {k}")
            print("Press Ctrl+C to disconnect and exit.")
            print("=" * 80 + "\n")

        # Keep running until cancelled
        while self._running:
            await asyncio.sleep(1)

    def _on_error(self, req_id: int, error_code: int, error_string: str, contract: Contract) -> None:
        """Handle errors and notifications from IB Gateway."""
        # Filter out informational notification codes (e.g. 2104, 2106, 2158)
        if error_code in (2104, 2106, 2158):
            return
        sym = getattr(contract, "symbol", "") if contract else ""
        print(f"[IBKR MSG {error_code}] (reqId={req_id}, symbol={sym}): {error_string}")

    async def _subscribe_bars(self, label: str, contract: Contract, what_to_show: str) -> None:
        """Qualify a contract and request keepUpToDate 1-minute historical bars."""
        try:
            print(f"Qualifying contract for {label}...")
            qualified = await asyncio.wait_for(self.ib.qualifyContractsAsync(contract), timeout=6.0)
            if not qualified:
                print(f"[WARN] Failed to qualify contract for {label}")
                return

            print(f"Subscribing to 1-minute bars for {label} (whatToShow={what_to_show})...")
            # Request historical data with keepUpToDate=True to receive continuous 1-minute updates
            bars = await asyncio.wait_for(
                self.ib.reqHistoricalDataAsync(
                    contract=contract,
                    endDateTime="",
                    durationStr="1800 S",  # Fetch last 30 minutes of bars initially
                    barSizeSetting="1 min",
                    whatToShow=what_to_show,
                    useRTH=self.use_rth,
                    formatDate=1,
                    keepUpToDate=True,
                ),
                timeout=7.0,
            )

            if not bars:
                print(f"[WARN] No initial bar data returned for {label}.")
                if "CRYPTO" in label:
                    print(
                        f"  -> Note: Paxos crypto on IB requires Paxos crypto permissions in Account Management. "
                        f"You can also track spot crypto ETFs via --stocks (e.g. --stocks AAPL,SPY,IBIT,FBTC)."
                    )
                # Keep subscription if it was created, else continue
                return

            self.active_subscriptions[label] = bars

            latest = bars[-1]
            print(
                f"[INITIAL] {label:<12} | Time: {latest.date} | "
                f"O: {latest.open:<8.2f} H: {latest.high:<8.2f} L: {latest.low:<8.2f} "
                f"C: {latest.close:<8.2f} Vol: {int(latest.volume):,}"
            )

            # Attach update callback
            def on_bar_update(bar_list: BarDataList, has_new_bar: bool, name: str = label):
                self._handle_bar_update(name, bar_list, has_new_bar)

            bars.updateEvent += on_bar_update

        except asyncio.TimeoutError:
            print(f"[TIMEOUT] Request for {label} bars timed out (often indicates missing market data subscription).")
            if "CRYPTO" in label:
                print(
                    f"  -> Tip: If you don't have PAXOS crypto data permissions on your IB account, "
                    f"you can track spot Bitcoin ETFs via --stocks (e.g. --stocks AAPL,SPY,IBIT,FBTC)."
                )
        except Exception as e:
            print(f"[ERROR] Could not subscribe to {label}: {e}")

    def _handle_bar_update(self, label: str, bars: BarDataList, has_new_bar: bool) -> None:
        """Invoked when a bar is updated or a new 1-minute bar closes."""
        if not bars:
            return

        current_bar = bars[-1]
        now_str = datetime.now().strftime("%H:%M:%S")

        if has_new_bar:
            # The previous 1-minute bar just finalized and closed!
            closed_bar = bars[-2] if len(bars) >= 2 else current_bar
            print(
                f"\n[{now_str}] >>> [1-MIN BAR CLOSED] {label:<12} <<<\n"
                f"  Bar Time : {closed_bar.date}\n"
                f"  Open     : {closed_bar.open:.2f}\n"
                f"  High     : {closed_bar.high:.2f}\n"
                f"  Low      : {closed_bar.low:.2f}\n"
                f"  Close    : {closed_bar.close:.2f}\n"
                f"  Volume   : {int(closed_bar.volume):,}\n"
                f"  WAP      : {closed_bar.average:.2f}\n"
                f"  Count    : {closed_bar.barCount}\n"
            )
        else:
            # In-progress 1-minute bar tick update
            print(
                f"[{now_str}] [TICK] {label:<12} | Time: {current_bar.date} | "
                f"O: {current_bar.open:.2f} H: {current_bar.high:.2f} L: {current_bar.low:.2f} "
                f"C: {current_bar.close:.2f} Vol: {int(current_bar.volume):,}"
            )

    def stop(self) -> None:
        """Cancel subscriptions and disconnect cleanly."""
        self._running = False
        print("\nStopping listener...")
        for label, bars in self.active_subscriptions.items():
            print(f"Cancelling bar subscription for {label}...")
            try:
                self.ib.cancelHistoricalData(bars)
            except Exception:
                pass
        self.active_subscriptions.clear()
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from IB Gateway.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream 1-minute OHLCV bars for Stocks and Crypto from IB Gateway."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="IB Gateway host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4002,
        help="IB Gateway socket port (4002 for paper, 4001 for live, 7497/7496 for TWS). Default: 4002",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=10,
        help="Unique client ID (default: 10)",
    )
    parser.add_argument(
        "--stocks",
        type=str,
        default="AAPL,SPY",
        help="Comma-separated stock symbols (e.g. AAPL,SPY,NVDA,IBIT)",
    )
    parser.add_argument(
        "--cryptos",
        type=str,
        default="BTC,ETH",
        help="Comma-separated crypto symbols (e.g. BTC,ETH)",
    )
    parser.add_argument(
        "--crypto-exchange",
        type=str,
        default="PAXOS",
        help="Crypto exchange (PAXOS or ZEROHASH, default: PAXOS)",
    )
    parser.add_argument(
        "--rth",
        action="store_true",
        help="Restrict stock bars to Regular Trading Hours (RTH) only (default: False)",
    )
    parser.add_argument(
        "--delayed",
        action="store_true",
        help="Request delayed market data if live subscription is not available (default: False)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stocks = [s for s in args.stocks.split(",") if s]
    cryptos = [c for c in args.cryptos.split(",") if c]

    listener = BarListener(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        stocks=stocks,
        cryptos=cryptos,
        crypto_exchange=args.crypto_exchange,
        use_rth=args.rth,
        delayed=args.delayed,
    )

    # Set up graceful shutdown on SIGINT / SIGTERM
    def shutdown():
        listener.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        loop.run_until_complete(listener.start())
    except (KeyboardInterrupt, SystemExit):
        listener.stop()
    finally:
        loop.close()


if __name__ == "__main__":
    main()

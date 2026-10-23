# Marimo generates cell-tail display expressions and explicit returns.
# ruff: noqa: B018, PLR1711

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    from datetime import datetime
    import math
    import time
    from typing import Any, Callable

    import marimo as mo
    import pandas as pd

    return Any, Callable, asyncio, datetime, math, mo, pd, time


@app.cell
def _(mo):
    get_market_state, set_market_state = mo.state(
        {
            "status": "Disconnected. Toggle the switch above to connect to IB Gateway.",
            "connected": False,
            "last_updated": "-",
            "instruments": {},
            "total_count": 0,
        }
    )
    return get_market_state, set_market_state


@app.cell
def _(Any, Callable, asyncio, datetime):
    def _load_ib():
        from ib_async import ContFuture, Contract, Crypto, IB, Stock

        return IB, ContFuture, Crypto, Stock

    class Level1FeedController:
        """Manages live Level 1 streaming subscriptions for Stocks, Cryptos, and Futures."""

        TOP_20_STOCKS = [
            "SPY",
            "QQQ",
            "NVDA",
            "TSLA",
            "AAPL",
            "AMZN",
            "MSFT",
            "AMD",
            "META",
            "GOOGL",
            "PLTR",
            "AVGO",
            "SMCI",
            "COIN",
            "IBIT",
            "IWM",
            "BAC",
            "INTC",
            "BABA",
            "MARA",
        ]

        def __init__(
            self,
            publish: Callable[[dict[str, Any]], None],
            host: str = "127.0.0.1",
            port: int = 4002,
            client_id: int = 33,
            market_data_type: int = 1,
            refresh_interval: float = 1.0,
        ):
            self.publish = publish
            self.host = host
            self.port = port
            self.client_id = client_id
            self.market_data_type = market_data_type
            self.refresh_interval = refresh_interval
            self.task: asyncio.Task | None = None
            self.ib = None
            self._connection = (self.host, self.port, self.market_data_type)

        def configure(self, port: int, market_data_type: int, refresh_interval: float):
            self.port = port
            self.market_data_type = market_data_type
            self.refresh_interval = refresh_interval

        async def set_enabled(self, enabled: bool):
            if enabled:
                current = (self.host, self.port, self.market_data_type)
                running = self.task is not None and not self.task.done()
                if running and current == self._connection:
                    return
                if running:
                    task = self.task
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    self.task = None
                self._connection = current
                self.task = asyncio.create_task(self._run(), name="ib-level1-feed")
                return
            if self.task is not None and not self.task.done():
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            self.task = None
            self.publish(
                {
                    "status": "Disconnected. Toggle switch to reconnect.",
                    "connected": False,
                    "last_updated": "-",
                    "instruments": {},
                    "total_count": 0,
                }
            )

        async def _run(self):
            IB, ContFuture, Crypto, Stock = _load_ib()
            ib = IB()
            self.ib = ib
            contracts_info: list[tuple[str, str, Any]] = []
            subscribed_contracts = []
            tickers = {}
            try:
                self.publish(
                    {
                        "status": f"Connecting to IB Gateway on {self.host}:{self.port}...",
                        "connected": False,
                        "last_updated": "-",
                        "instruments": {},
                        "total_count": 0,
                    }
                )

                # Auto-assign unique client ID to prevent collisions
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
                    raise RuntimeError(
                        "Unable to connect to IB Gateway (all client IDs in use)"
                    )

                ib.reqMarketDataType(self.market_data_type)

                # Build contract list
                cl = ContFuture("CL", "NYMEX")
                gc = ContFuture("GC", "COMEX")
                contracts_info.append(("CL", "Commodity (Crude Oil)", cl))
                contracts_info.append(("GC", "Commodity (Gold)", gc))

                btc = Crypto("BTC", "PAXOS", "USD")
                eth = Crypto("ETH", "PAXOS", "USD")
                contracts_info.append(("BTC", "Crypto (Bitcoin)", btc))
                contracts_info.append(("ETH", "Crypto (Ethereum)", eth))

                for symbol in self.TOP_20_STOCKS:
                    contracts_info.append(
                        (symbol, "US Equity", Stock(symbol, "SMART", "USD"))
                    )

                contracts = [contract for _, _, contract in contracts_info]
                await ib.qualifyContractsAsync(*contracts)

                for symbol, category, contract in contracts_info:
                    ticker = ib.reqMktData(contract)
                    subscribed_contracts.append(contract)
                    tickers[symbol] = (category, ticker)

                md_label = "Live" if self.market_data_type == 1 else "Delayed (~15 min)"
                self.publish(
                    {
                        "status": f"🟢 Connected to IB Gateway ({self.port}) | Streaming {len(tickers)} instruments [{md_label}]",
                        "connected": True,
                        "last_updated": datetime.now().strftime("%H:%M:%S"),
                        "instruments": {},
                        "total_count": len(tickers),
                    }
                )

                while True:
                    await asyncio.sleep(self.refresh_interval)
                    now_str = datetime.now().strftime("%H:%M:%S")
                    snapshot_data = {}
                    for symbol, (category, ticker) in tickers.items():
                        bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
                        ask = ticker.ask if ticker.ask and ticker.ask > 0 else None
                        bid_size = (
                            ticker.bidSize
                            if ticker.bidSize and ticker.bidSize > 0
                            else None
                        )
                        ask_size = (
                            ticker.askSize
                            if ticker.askSize and ticker.askSize > 0
                            else None
                        )
                        last = (
                            ticker.last
                            if ticker.last and ticker.last > 0
                            else (bid or ask)
                        )
                        close = (
                            ticker.close if ticker.close and ticker.close > 0 else None
                        )
                        high = ticker.high if ticker.high and ticker.high > 0 else None
                        low = ticker.low if ticker.low and ticker.low > 0 else None
                        volume = (
                            ticker.volume
                            if ticker.volume and ticker.volume > 0
                            else None
                        )

                        spread = round(ask - bid, 4) if (bid and ask) else None
                        pct_change = (
                            round(((last - close) / close) * 100, 2)
                            if (last and close)
                            else None
                        )
                        snapshot_data[symbol] = {
                            "Symbol": symbol,
                            "Category": category,
                            "Bid Size": bid_size,
                            "Bid": bid,
                            "Ask": ask,
                            "Ask Size": ask_size,
                            "Spread": spread,
                            "Last": last,
                            "Day High": high,
                            "Day Low": low,
                            "Close": close,
                            "Change %": pct_change,
                            "Volume": volume,
                        }

                    self.publish(
                        {
                            "status": f"🟢 Connected to IB Gateway ({self.port}) | Streaming {len(tickers)} instruments [{md_label}]",
                            "connected": True,
                            "last_updated": now_str,
                            "instruments": snapshot_data,
                            "total_count": len(tickers),
                        }
                    )
            finally:
                for contract in subscribed_contracts:
                    try:
                        ib.cancelMktData(contract)
                    except Exception:
                        pass
                try:
                    if ib.isConnected():
                        ib.disconnect()
                finally:
                    if self.ib is ib:
                        self.ib = None

    return Level1FeedController, _load_ib


@app.cell
def _(Level1FeedController, set_market_state):
    controller = Level1FeedController(set_market_state, host="127.0.0.1", port=4002)
    return (controller,)


@app.cell
def _(mo):
    connect_switch = mo.ui.switch(
        value=False,
        label="⚡ Connect Live Stream",
    )
    port_dropdown = mo.ui.dropdown(
        options={
            "4002 (Paper Gateway)": 4002,
            "4001 (Live Gateway)": 4001,
            "7497 (Paper TWS)": 7497,
            "7496 (Live TWS)": 7496,
        },
        value="4002 (Paper Gateway)",
        label="Gateway Port",
    )
    market_data_dropdown = mo.ui.dropdown(
        options={"Live Real-Time": 1, "Delayed (~15 min)": 3},
        value="Live Real-Time",
        label="Market Data Type",
    )
    refresh_rate_slider = mo.ui.slider(
        start=0.5,
        stop=3.0,
        step=0.5,
        value=1.0,
        label="UI Refresh Rate (sec)",
    )

    return (
        connect_switch,
        market_data_dropdown,
        port_dropdown,
        refresh_rate_slider,
    )


@app.cell
def _(
    controller,
    market_data_dropdown,
    port_dropdown,
    refresh_rate_slider,
):
    controller.configure(
        port=int(port_dropdown.value),
        market_data_type=int(market_data_dropdown.value),
        refresh_interval=float(refresh_rate_slider.value),
    )
    return


@app.cell
async def _(connect_switch, controller):
    await controller.set_enabled(connect_switch.value)
    return


@app.cell
def _(
    connect_switch,
    get_market_state,
    market_data_dropdown,
    mo,
    port_dropdown,
    refresh_rate_slider,
):
    state = get_market_state()
    status_msg = state.get("status", "")
    last_up = state.get("last_updated", "-")

    control_bar = mo.hstack(
        [
            connect_switch,
            port_dropdown,
            market_data_dropdown,
            refresh_rate_slider,
        ],
        justify="start",
        gap=2,
    )

    banner = mo.md(f"""
    # 📈 Real-Time Level 1 Market Watch
    **Streaming Top 20 Traded US Stocks, Cryptos (BTC/ETH), and Futures (CL Crude Oil & GC Gold) via IB Gateway Socket.**

    Status: **{status_msg}** &nbsp;|&nbsp; Last Update: `{last_up}`
    """)

    mo.vstack([banner, control_bar], gap=1)
    return banner, control_bar, last_up, state, status_msg


@app.cell
def _(get_market_state, mo):
    data = get_market_state().get("instruments", {})

    def _render_card(sym: str, title: str):
        item = data.get(sym, {})
        last = item.get("Last")
        chg = item.get("Change %")
        bid = item.get("Bid")
        ask = item.get("Ask")
        high = item.get("Day High")
        low = item.get("Day Low")

        price_str = f"${last:,.2f}" if last is not None else "--"
        caption = ""
        if bid is not None and ask is not None:
            caption = f"B: {bid:,.2f} | A: {ask:,.2f}"
            if high and low:
                caption += f" | Range: {low:,.2f} - {high:,.2f}"

        direction = (
            "increase"
            if (chg and chg > 0)
            else ("decrease" if (chg and chg < 0) else None)
        )
        return mo.stat(
            value=price_str,
            label=f"{title} ({sym})",
            caption=caption or "Awaiting stream...",
            direction=direction,
        )

    kpi_cards = mo.hstack(
        [
            _render_card("CL", "Crude Oil (WTI)"),
            _render_card("GC", "Gold (COMEX)"),
            _render_card("BTC", "Bitcoin (PAXOS)"),
            _render_card("ETH", "Ethereum (PAXOS)"),
        ],
        justify="space-between",
        gap=2,
    )
    kpi_cards
    return (kpi_cards,)


@app.cell
def _(get_market_state, mo, pd):
    raw_data = get_market_state().get("instruments", {})
    if not raw_data:
        display_out = mo.md(
            "_No active market data streaming. Switch on the feed above to view quotes._"
        )
    else:
        df_all = pd.DataFrame(list(raw_data.values()))

        # Separate datasets
        df_stocks = df_all[df_all["Category"] == "US Equity"].copy()
        df_futures = df_all[df_all["Category"].str.contains("Commodity")].copy()
        df_crypto = df_all[df_all["Category"].str.contains("Crypto")].copy()

        def _format_df(df):
            if df.empty:
                return df
            return df[
                [
                    "Symbol",
                    "Category",
                    "Last",
                    "Change %",
                    "Bid Size",
                    "Bid",
                    "Ask",
                    "Ask Size",
                    "Spread",
                    "Day High",
                    "Day Low",
                    "Volume",
                ]
            ]

        display_out = mo.ui.tabs(
            {
                "⭐ All Instruments": mo.ui.table(
                    _format_df(df_all), pagination=True, page_size=25
                ),
                "🏛️ Top 20 Stocks": mo.ui.table(
                    _format_df(df_stocks), pagination=True, page_size=20
                ),
                "🛢️ Futures (CL & GC)": mo.ui.table(
                    _format_df(df_futures), pagination=False
                ),
                "🪙 Crypto (BTC & ETH)": mo.ui.table(
                    _format_df(df_crypto), pagination=False
                ),
            }
        )

    display_out
    return (display_out,)


if __name__ == "__main__":
    app.run()

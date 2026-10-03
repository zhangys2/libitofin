# Marimo generates cell-tail display expressions and explicit returns.
# ruff: noqa: B018, PLR1711

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import textwrap
    import time

    import marimo as mo
    import pandas as pd
    from btc_option_iv.kalman import FilterConfig
    from btc_option_iv.live import LiveFeedController
    from btc_option_iv.reference import butterfly_markdown
    from btc_option_iv.views import (
        make_quotes_dataframe,
        make_smile_figure,
        make_smile_reader,
        make_surface_figure,
    )
    from spx_option_iv.feed import SpxFeedController
    from spx_option_iv.views import make_spx_smile_figure, make_spx_smile_reader

    return (
        FilterConfig,
        LiveFeedController,
        SpxFeedController,
        butterfly_markdown,
        make_quotes_dataframe,
        make_smile_figure,
        make_smile_reader,
        make_spx_smile_figure,
        make_spx_smile_reader,
        make_surface_figure,
        mo,
        pd,
        textwrap,
        time,
    )


@app.cell
def _(mo):
    get_spx_snapshot, set_spx_snapshot = mo.state(
        {
            "status": "Feed stopped. Switch on to connect to IB Gateway.",
            "connected": False,
            "spot_price": None,
            "expiry_str": None,
            "expiry_timestamp_ms": None,
            "time_to_close_hours": None,
            "latest_quotes": {},
            "kalman_views": {},
            "feed_running": False,
            "published_ms": 0,
        }
    )

    get_btc_snapshot, set_btc_snapshot = mo.state(
        {
            "status": "Feed stopped. Switch on to connect to Deribit.",
            "latest_options": {},
            "instruments": {},
            "rows": {},
            "book": None,
            "kalman_views": {},
            "feed_running": False,
        }
    )

    get_btc_expiry_catalog, set_btc_expiry_catalog = mo.state(())
    get_btc_chosen_expiry, set_btc_chosen_expiry = mo.state(None)
    return (
        get_btc_chosen_expiry,
        get_btc_expiry_catalog,
        get_btc_snapshot,
        get_spx_snapshot,
        set_btc_chosen_expiry,
        set_btc_expiry_catalog,
        set_btc_snapshot,
        set_spx_snapshot,
    )


@app.cell
def _(
    LiveFeedController,
    SpxFeedController,
    set_btc_snapshot,
    set_spx_snapshot,
):
    spx_controller = SpxFeedController(set_spx_snapshot, host="127.0.0.1", port=4002)
    btc_controller = LiveFeedController(set_btc_snapshot)
    return btc_controller, spx_controller


@app.cell
def _(mo):
    mo.md("""
    # ⚡ Real-Time Volatility Surface & Smile Platform
    Simultaneous calibration of **SPX 0DTE Index Options** (via **Interactive Brokers TCP Socket**) and
    **Bitcoin Options Surface** (via **Deribit WebSocket**).
    Features **Bayesian linear-Gaussian Kalman filtering** across 9 standardized moneyness knots,
    **Total Variance Cubic Smile interpolation** with **Roger Lee wing asymptotics**, and
    **Durrleman butterfly arbitrage verification**.
    """)
    return


@app.cell
def _(mo):
    view_mode = mo.ui.radio(
        options=["📊 Both (Split View)", "⚡ SPX 0DTE Only", "🪙 BTC Surface Only"],
        value="📊 Both (Split View)",
        label="Dashboard View:",
    )
    view_mode
    return (view_mode,)


@app.cell
def _(mo):
    spx_connect = mo.ui.switch(
        value=False, label="⚡ Connect SPX 0DTE (IBKR Socket)"
    )
    btc_connect = mo.ui.switch(
        value=False, label="🪙 Connect BTC Surface (Deribit WS)"
    )
    spx_port = mo.ui.dropdown(
        options={"4002 (Paper)": 4002, "4001 (Live)": 4001},
        value="4002 (Paper)",
        label="Gateway Port",
    )
    spx_data_type = mo.ui.dropdown(
        options={"Live (OPRA subscription)": 1, "Delayed (~15 min, free)": 3},
        value="Live (OPRA subscription)",
        label="Market Data (applies on connect)",
    )
    spx_strike_radius = mo.ui.slider(
        start=30,
        stop=160,
        step=10,
        value=80,
        label="SPX Strike Window (±pts)",
    )
    spx_process_rate = mo.ui.slider(
        start=0.01,
        stop=2.0,
        step=0.01,
        value=0.10,
        label="SPX Process IV (pts / √sec)",
    )
    spx_meas_sd = mo.ui.slider(
        start=0.05,
        stop=5.0,
        step=0.05,
        value=0.50,
        label="SPX Meas SD (pts)",
    )
    spx_reset = mo.ui.button(
        value=0, on_click=lambda v: v + 1, label="Reset SPX Filter"
    )

    btc_process_rate = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.01,
        value=0.10,
        label="BTC Process IV (% pts / √sec)",
    )
    btc_meas_sd = mo.ui.slider(
        start=0.01,
        stop=20.0,
        step=0.01,
        value=0.50,
        label="BTC Meas SD (% pts)",
    )
    btc_reset = mo.ui.button(
        value=0, on_click=lambda v: v + 1, label="Reset BTC Filter"
    )
    return (
        btc_connect,
        btc_meas_sd,
        btc_process_rate,
        btc_reset,
        spx_connect,
        spx_data_type,
        spx_meas_sd,
        spx_port,
        spx_process_rate,
        spx_reset,
        spx_strike_radius,
    )


@app.cell
def _(
    FilterConfig,
    spx_controller,
    spx_data_type,
    spx_meas_sd,
    spx_port,
    spx_process_rate,
    spx_strike_radius,
):
    spx_controller.set_port(int(spx_port.value))
    spx_controller.set_market_data_type(int(spx_data_type.value))
    spx_controller.set_strike_radius(float(spx_strike_radius.value))
    spx_controller.configure(
        FilterConfig(
            process_iv_rate=spx_process_rate.value / 100.0,
            measurement_sd=spx_meas_sd.value / 100.0,
        )
    )
    return


@app.cell
def _(spx_controller, spx_reset):
    if spx_reset.value:
        spx_controller.reset_filters()
    return


@app.cell
async def _(spx_connect, spx_controller):
    await spx_controller.set_enabled(spx_connect.value)
    return


@app.cell
def _(FilterConfig, btc_controller, btc_meas_sd, btc_process_rate):
    btc_controller.configure(
        FilterConfig(
            process_iv_rate=btc_process_rate.value / 100.0,
            measurement_sd=btc_meas_sd.value / 100.0,
        )
    )
    return


@app.cell
def _(btc_controller, btc_reset):
    if btc_reset.value:
        btc_controller.reset_filters()
    return


@app.cell
async def _(btc_connect, btc_controller):
    await btc_controller.set_enabled(btc_connect.value)
    return


@app.cell
def _(
    btc_connect,
    btc_meas_sd,
    btc_process_rate,
    btc_reset,
    mo,
    spx_connect,
    spx_data_type,
    spx_meas_sd,
    spx_port,
    spx_process_rate,
    spx_reset,
    spx_strike_radius,
    view_mode,
):
    header_controls = mo.hstack(
        [spx_connect, btc_connect],
        justify="start",
        gap=4,
    )

    if view_mode.value == "⚡ SPX 0DTE Only":
        tuning_ui = mo.vstack(
            [
                mo.hstack([spx_port, spx_data_type, spx_strike_radius], justify="start", gap=2),
                mo.hstack([spx_process_rate, spx_meas_sd, spx_reset], justify="start", gap=2),
            ],
            gap=1,
        )
    elif view_mode.value == "🪙 BTC Surface Only":
        tuning_ui = mo.hstack([btc_process_rate, btc_meas_sd, btc_reset], justify="start", gap=2)
    else:
        tuning_ui = mo.vstack(
            [
                mo.md("#### Tuning Controls"),
                mo.hstack([spx_port, spx_data_type, spx_strike_radius, spx_process_rate, spx_meas_sd, spx_reset], justify="start", gap=2),
                mo.hstack([btc_process_rate, btc_meas_sd, btc_reset], justify="start", gap=2),
            ],
            gap=1,
        )

    controls_panel = mo.vstack([header_controls, tuning_ui], gap=2)
    controls_panel
    return


@app.cell
def _(get_spx_snapshot, make_spx_smile_reader):
    spx_snapshot = get_spx_snapshot()
    spx_analysis, spx_error = make_spx_smile_reader(spx_snapshot)
    return spx_analysis, spx_error, spx_snapshot


@app.cell
def _(get_btc_expiry_catalog, get_btc_snapshot, set_btc_expiry_catalog):
    btc_snapshot = get_btc_snapshot()
    # Discover expiries
    _avail = tuple(
        sorted(
            {
                inst.expiration_timestamp_ms
                for inst in btc_snapshot["instruments"].values()
            }
        )
    )
    if _avail != get_btc_expiry_catalog():
        set_btc_expiry_catalog(_avail)
    return (btc_snapshot,)


@app.cell
def _(
    get_btc_chosen_expiry,
    get_btc_expiry_catalog,
    mo,
    set_btc_chosen_expiry,
    time,
):
    expiry_timestamps = get_btc_expiry_catalog()
    expiry_options = {
        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(e / 1000.0)): e
        for e in expiry_timestamps
    }

    if expiry_options:
        btc_expiry_selector = mo.ui.dropdown(
            options=expiry_options,
            value=next(
                (
                    lbl
                    for lbl, e in expiry_options.items()
                    if e == get_btc_chosen_expiry()
                ),
                next(iter(expiry_options)),
            ),
            on_change=set_btc_chosen_expiry,
            label="Expiry for BTC Cubic Smile Reader",
        )
    else:
        btc_expiry_selector = None
    return (btc_expiry_selector,)


@app.cell
def _(btc_expiry_selector):
    selected_expiry_ms = (
        btc_expiry_selector.value if btc_expiry_selector is not None else None
    )
    return (selected_expiry_ms,)


@app.cell
def _(btc_snapshot, make_smile_reader, selected_expiry_ms):
    btc_analysis, btc_error = make_smile_reader(btc_snapshot, selected_expiry_ms)
    return btc_analysis, btc_error


@app.cell
def _(
    btc_analysis,
    btc_error,
    btc_expiry_selector,
    btc_snapshot,
    butterfly_markdown,
    make_quotes_dataframe,
    make_smile_figure,
    make_spx_smile_figure,
    make_surface_figure,
    mo,
    pd,
    spx_analysis,
    spx_error,
    spx_snapshot,
    textwrap,
    time,
    view_mode,
):
    # ---------------------------------------------------------------------
    # BUILD SPX COMPONENT
    # ---------------------------------------------------------------------
    if spx_error or spx_analysis is None:
        spx_status_md = mo.md(f"**SPX Status:** {spx_snapshot.get('status', 'Waiting for SPX data...')}")
    else:
        view = spx_analysis["view"]
        butterfly_md = butterfly_markdown(
            spx_analysis.get("butterfly_report"), label="Reference smile (total variance) butterfly check"
        )
        if butterfly_md:
            butterfly_md += "<br>"

        hours = spx_analysis["time_to_close_hours"]
        h_int = int(hours)
        m_int = int((hours % 1) * 60)
        time_left_str = f"{h_int}h {m_int:02d}m"

        spx_status_md = mo.md(
            textwrap.dedent(f"""
            ### ⚡ SPX 0DTE Volatility Smile (IBKR Socket)
            **SPX Spot / Forward:** `{spx_analysis['spot_price']:,.2f} USD` · **ATM Vol:** `{spx_analysis['atm_vol']:.2%}` · **0DTE Expiration:** `{spx_analysis['expiry_str']}` (`{time_left_str}` to 4:00 PM ET close)<br>
            {butterfly_md}
            **Accepted Updates:** `{view.updates}` · **Resets:** `{view.resets}` · **Filter State:** `{view.status}` · **Last Reset Reason:** `{view.reset_reason}`<br>
            _The arbitrage check applies to the dashed reference smile only. The solid Kalman curve is a natural cubic in IV: not arbitrage-free, not a trading signal._
            """)
        )

    spx_figure = make_spx_smile_figure(spx_analysis, spx_error)

    if spx_analysis and "observations" in spx_analysis:
        obs_df = spx_analysis["observations"].copy()
        knots_df = spx_analysis["knots"].copy()

        def fmt_pct(val):
            return f"{val:.2%}" if pd.notna(val) else ""

        def fmt_usd(val):
            return f"{val:,.2f}" if pd.notna(val) else ""

        if not obs_df.empty:
            for col in ["observed_mid_iv", "fitted_iv", "filtered_iv"]:
                if col in obs_df:
                    obs_df[col] = obs_df[col].apply(fmt_pct)
            for col in ["fitted_minus_observed", "filtered_minus_observed", "innovation"]:
                if col in obs_df:
                    obs_df[col] = obs_df[col].apply(
                        lambda v: f"{v * 100:+.2f} pts" if pd.notna(v) else ""
                    )
            for col in ["bid_usd", "ask_usd", "mid_usd", "spread_usd"]:
                if col in obs_df:
                    obs_df[col] = obs_df[col].apply(fmt_usd)

        for col in ["filtered_iv", "reference_iv"]:
            if col in knots_df:
                knots_df[col] = knots_df[col].apply(fmt_pct)

        spx_tables_ui = mo.ui.tabs(
            {
                "📊 SPX 0DTE Quotes & Residuals": mo.ui.table(
                    obs_df, pagination=True, page_size=10
                ),
                "📍 SPX Kalman Spline Knots (9 Knots)": mo.ui.table(
                    knots_df, pagination=False
                ),
            }
        )
    else:
        spx_tables_ui = mo.md("_No active SPX option quotes to display._")

    spx_section = mo.vstack(
        [
            spx_status_md,
            mo.ui.plotly(spx_figure),
            spx_tables_ui,
        ],
        gap=2,
    )

    # ---------------------------------------------------------------------
    # BUILD BTC COMPONENT
    # ---------------------------------------------------------------------
    btc_status_md = mo.md(f"""
    ### 🪙 Bitcoin Volatility Surface & Smile (Deribit WebSocket)
    **Feed Status:** {btc_snapshot.get('status', 'Feed stopped.')}
    """)

    surface_fig = make_surface_figure(btc_snapshot.get("rows", {}))

    if btc_analysis is None:
        btc_smile_md = mo.md(
            f"**Cubic Smile Reader:** {btc_error or 'Start feed to discover expiries.'}"
        )
        btc_smile_fig = make_smile_figure(None, btc_error)
        btc_tables_ui = mo.md("_No active expiry selected._")
    else:
        _view = btc_analysis["view"]
        _exp_label = time.strftime(
            "%Y-%m-%d %H:%M UTC",
            time.gmtime(btc_analysis["expiry_timestamp_ms"] / 1000.0),
        )
        _age = btc_analysis["last_accepted_age"]
        _age_text = f"{_age:.1f}s" if _age is not None else "not initialized"

        _butterfly_text = butterfly_markdown(btc_analysis.get("butterfly_report"))
        if _butterfly_text:
            _butterfly_text = "<br>" + _butterfly_text

        btc_smile_md = mo.md(
            textwrap.dedent(f"""
            #### Selected Expiry Smile: `{_exp_label}`
            **Filter Status:** {btc_analysis['filter_status']} · **Last accepted update:** {_age_text}<br>
            **Forward:** {btc_analysis['forward']:,.2f} USD · **Expiry:** {btc_analysis['exercise_time']:.8f} years · **ATM Vol:** {btc_analysis['atm_vol']:.2%}<br>
            **Accepted updates:** {_view.updates} · **Resets:** {_view.resets} · **Last reset:** {_view.reset_reason}
            {_butterfly_text}
            """)
        )
        btc_smile_fig = make_smile_figure(btc_analysis, btc_error)

        btc_quotes_df = make_quotes_dataframe(btc_snapshot)
        btc_knots_df = btc_analysis.get("knots", pd.DataFrame())
        btc_obs_df = btc_analysis.get("observations", pd.DataFrame())

        btc_tables_ui = mo.ui.tabs(
            {
                "⚡ 100 Most Recent BTC Quotes": mo.ui.table(
                    btc_quotes_df, pagination=True, page_size=10
                ),
                "📊 Selected Expiry Residuals": mo.ui.table(
                    btc_obs_df, pagination=True, page_size=10
                ),
                "📍 BTC Filtered Knots (9 Knots)": mo.ui.table(
                    btc_knots_df, pagination=False
                ),
            }
        )

    selector_ui = btc_expiry_selector if btc_expiry_selector is not None else mo.md("_Start feed to populate expiries._")

    btc_section = mo.vstack(
        [
            btc_status_md,
            mo.ui.plotly(surface_fig),
            selector_ui,
            btc_smile_md,
            mo.ui.plotly(btc_smile_fig),
            btc_tables_ui,
        ],
        gap=2,
    )

    # ---------------------------------------------------------------------
    # LAYOUT ACCORDING TO VIEW MODE
    # ---------------------------------------------------------------------
    if view_mode.value == "⚡ SPX 0DTE Only":
        final_layout = spx_section
    elif view_mode.value == "🪙 BTC Surface Only":
        final_layout = btc_section
    else:
        # Both (Split / Stacked View)
        final_layout = mo.vstack(
            [
                spx_section,
                mo.md("---"),
                btc_section,
            ],
            gap=3,
        )

    final_layout
    return


if __name__ == "__main__":
    app.run()

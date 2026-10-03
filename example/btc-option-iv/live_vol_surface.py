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

    return (
        FilterConfig,
        LiveFeedController,
        butterfly_markdown,
        make_quotes_dataframe,
        make_smile_figure,
        make_smile_reader,
        make_surface_figure,
        mo,
        pd,
        textwrap,
        time,
    )


@app.cell
def _(mo):
    get_market_snapshot, set_market_snapshot = mo.state(
        {
            "status": "Feed stopped. Switch it on to connect.",
            "latest_options": {},
            "instruments": {},
            "rows": {},
            "book": None,
            "kalman_views": {},
            "feed_running": False,
        }
    )
    return get_market_snapshot, set_market_snapshot


@app.cell
def _(LiveFeedController, set_market_snapshot):
    feed_controller = LiveFeedController(set_market_snapshot)
    return (feed_controller,)


@app.cell
def _(mo):
    live_enabled = mo.ui.switch(value=False, label="Connect to live Deribit quotes")
    live_enabled
    return (live_enabled,)


@app.cell
async def _(feed_controller, live_enabled):
    await feed_controller.set_enabled(live_enabled.value)
    return


@app.cell
def _(mo):
    process_rate = mo.ui.number(
        start=0.0,
        stop=10.0,
        step=0.01,
        value=0.1,
        label="Process IV rate (percentage points / √second)",
    )
    measurement_sd = mo.ui.number(
        start=0.01,
        stop=20.0,
        step=0.01,
        value=0.5,
        label="Measurement SD at a 1-point IV spread (percentage points)",
    )
    reset_filters = mo.ui.button(
        value=0, on_click=lambda value: value + 1, label="Reset filters"
    )
    mo.hstack([process_rate, measurement_sd, reset_filters])
    return measurement_sd, process_rate, reset_filters


@app.cell
def _(FilterConfig, feed_controller, measurement_sd, process_rate):
    feed_controller.configure(
        FilterConfig(
            process_iv_rate=process_rate.value / 100.0,
            measurement_sd=measurement_sd.value / 100.0,
        )
    )
    return


@app.cell
def _(feed_controller, reset_filters):
    if reset_filters.value:
        feed_controller.reset_filters()
    return


@app.cell
def _(get_market_snapshot):
    market_snapshot = get_market_snapshot()
    return (market_snapshot,)


@app.cell
def _(market_snapshot, mo):
    mo.md(f"""
    **Feed status:** {market_snapshot["status"]}
    """)
    return


@app.cell
def _(make_surface_figure, market_snapshot, mo):
    surface_figure = make_surface_figure(market_snapshot["rows"])
    mo.ui.plotly(surface_figure)
    return


@app.cell
def _(make_quotes_dataframe, market_snapshot):
    make_quotes_dataframe(market_snapshot)
    return


@app.cell
def _(mo):
    get_expiry_catalog, set_expiry_catalog = mo.state(())
    get_chosen_expiry, set_chosen_expiry = mo.state(None)
    return get_expiry_catalog, set_expiry_catalog, get_chosen_expiry, set_chosen_expiry


@app.cell
def _(get_expiry_catalog, market_snapshot, set_expiry_catalog):
    # Quotes refresh every three seconds; the dropdown must not. Publish only
    # actual expiry-list changes to its separate reactive state.
    _available_expiries = tuple(
        sorted(
            {
                instrument.expiration_timestamp_ms
                for instrument in market_snapshot["instruments"].values()
            }
        )
    )
    if _available_expiries != get_expiry_catalog():
        set_expiry_catalog(_available_expiries)
    return


@app.cell
def _(get_chosen_expiry, get_expiry_catalog, mo, set_chosen_expiry, time):
    expiry_timestamps = get_expiry_catalog()
    expiry_options = {
        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(expiry_ms / 1000.0)): expiry_ms
        for expiry_ms in expiry_timestamps
    }
    if expiry_options:
        expiry_selector = mo.ui.dropdown(
            options=expiry_options,
            value=next(
                (
                    label
                    for label, expiry_ms in expiry_options.items()
                    if expiry_ms == get_chosen_expiry()
                ),
                next(iter(expiry_options)),
            ),
            on_change=set_chosen_expiry,
            label="Expiry for cubic smile reader",
        )
        selector_output = expiry_selector
    else:
        expiry_selector = None
        selector_output = mo.md(
            "Start the live feed to discover expiries for the cubic smile reader."
        )
    selector_output
    return (expiry_selector,)


@app.cell
def _(expiry_selector):
    selected_expiry_ms = expiry_selector.value if expiry_selector is not None else None
    return (selected_expiry_ms,)


@app.cell
def _(make_smile_reader, market_snapshot, selected_expiry_ms):
    smile_analysis, smile_error = make_smile_reader(market_snapshot, selected_expiry_ms)
    return smile_analysis, smile_error


@app.cell
def _(butterfly_markdown, mo, smile_analysis, smile_error, textwrap, time):
    if smile_analysis is None:
        summary_output = mo.md(f"### Selected-expiry Kalman smile\n\n{smile_error}")
    else:
        _view = smile_analysis["view"]
        _expiry_label = time.strftime(
            "%Y-%m-%d %H:%M UTC",
            time.gmtime(smile_analysis["expiry_timestamp_ms"] / 1000.0),
        )
        _age = smile_analysis["last_accepted_age"]
        _age_text = f"{_age:.1f}s" if _age is not None else "not initialized"
        _warnings = list(_view.warnings)
        if _view.reference_error:
            _warnings.append(
                f"Current reference fit unavailable: {_view.reference_error}"
            )
        _warning_text = " · ".join(_warnings) or "None"
        _butterfly_text = butterfly_markdown(smile_analysis.get("butterfly_report"))
        if _butterfly_text:
            _butterfly_text = "<br>" + _butterfly_text
        summary_output = mo.md(
            textwrap.dedent(f"""
            ### Selected-expiry cubic smile reader — Kalman prototype

            **Selected expiry:** {_expiry_label}<br>
            **Status:** {smile_analysis["filter_status"]} · **Last accepted update:** {_age_text}<br>
            **Forward:** {smile_analysis["forward"]:,.2f} USD · **Expiry:** {smile_analysis["exercise_time"]:.8f} years<br>
            **ATM normalization IV:** {smile_analysis["atm_vol"]:.2%}<br>
            **Accepted measurement updates:** {_view.updates} · **Resets:** {_view.resets} · **Last reset:** {_view.reset_reason}<br>
            **Warnings:** {_warning_text}{_butterfly_text}

            Both curves use the same nine fixed knots. The dashed curve is the
            total variance cubic smile with Roger Lee wing asymptotics and Durrleman
            butterfly arbitrage verification; the solid curve filters raw mid IVs
            with inverse bid–ask IV-spread precision. Curvature regularization is
            used for bootstrap/reference, not repeatedly imposed on the filter.
            Coefficients and knot values below describe the **unclipped filtered**
            natural cubic. Negative plotted IVs are floored for display only.
            Sparse regions are model-dependent; covariance is not calibrated confidence.
        """)
        )
    summary_output
    return


@app.cell
def _(make_smile_figure, mo, smile_analysis, smile_error):
    smile_figure = make_smile_figure(smile_analysis, smile_error)
    mo.ui.plotly(smile_figure)
    return


@app.cell
def _(mo):
    mo.md("""
    #### Nine filtered knot IVs (internal, before display flooring)
    """)
    return


@app.cell
def _(pd, smile_analysis):
    knot_table = smile_analysis["knots"] if smile_analysis else pd.DataFrame()
    knot_table
    return


@app.cell
def _(mo):
    mo.md("""
    #### Filtered per-segment coefficients `(a, b, c)` — `σ_i + a·dx + b·dx² + c·dx³`
    """)
    return


@app.cell
def _(pd, smile_analysis):
    coefficient_table = (
        smile_analysis["coefficients"] if smile_analysis else pd.DataFrame()
    )
    coefficient_table
    return


@app.cell
def _(mo):
    mo.md("""
    #### Selected quotes: IV spreads, residuals, innovations, and measurement status
    """)
    return


@app.cell
def _(pd, smile_analysis):
    observation_table = (
        smile_analysis["observations"] if smile_analysis else pd.DataFrame()
    )
    observation_table
    return


if __name__ == "__main__":
    app.run()

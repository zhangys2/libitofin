"""Plotly figures, table readers, and analysis views for SPX 0DTE volatility smile."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from btc_option_iv.kalman import KNOTS, SmileSnapshot
from btc_option_iv.reference import butterfly_report, reference_vol


def make_spx_smile_reader(snapshot: dict):
    """Read an immutable SPX snapshot; never mutate or advance filter state."""
    kalman_views = snapshot.get("kalman_views", {})
    if not kalman_views:
        return None, snapshot.get("status", "Waiting for SPX 0DTE options data...")

    # Only the subscribed session's filter is shown; never an arbitrary expiry.
    expiry_ms = snapshot.get("expiry_timestamp_ms")
    view = kalman_views.get(expiry_ms)
    if view is None:
        return None, snapshot.get("status", "Waiting for the subscribed SPX expiry...")
    if view.knot_ivs is None or view.context is None:
        return None, "Kalman filter is waiting for sufficient two-sided quotes to bootstrap..."

    context = view.context
    spot = snapshot.get("spot_price", context.forward)
    hours_left = snapshot.get("time_to_close_hours", 0.0)

    latest_quotes = snapshot.get("latest_quotes", {})

    observation_rows = []
    diagnostics_by_inst = {d.instrument: d for d in view.diagnostics}

    for o in view.observations:
        x = float(context.coordinates([o.strike])[0])
        in_range = -3.0 <= x <= 3.0
        filtered_iv = float(view.volatility([x], display=False)[0]) if in_range else None

        reference_iv = reference_vol(view, o.strike) if in_range else None

        diag = diagnostics_by_inst.get(o.instrument)
        quote = latest_quotes.get(o.instrument)

        bid_usd = quote.bid_usd if quote else None
        ask_usd = quote.ask_usd if quote else None
        mid_usd = (bid_usd + ask_usd) / 2.0 if bid_usd and ask_usd else None
        spread_usd = ask_usd - bid_usd if bid_usd and ask_usd else None
        right = quote.right if quote else ("Call" if o.strike >= spot else "Put")

        observation_rows.append(
            {
                "strike_usd": o.strike,
                "right": right,
                "bid_usd": bid_usd,
                "ask_usd": ask_usd,
                "mid_usd": mid_usd,
                "spread_usd": spread_usd,
                "x_std_dev": x,
                "observed_mid_iv": o.mid_iv,
                "iv_spread": o.spread,
                "fitted_iv": reference_iv,
                "filtered_iv": filtered_iv,
                "fitted_minus_observed": (reference_iv - o.mid_iv)
                if reference_iv is not None
                else None,
                "filtered_minus_observed": (filtered_iv - o.mid_iv)
                if filtered_iv is not None
                else None,
                "innovation": diag.innovation if diag else None,
                "innovation_z": diag.innovation_z if diag else None,
                "measurement_status": diag.status if diag else "accepted",
                "used_in_fit": in_range,
            }
        )

    observations_df = (
        pd.DataFrame.from_records(observation_rows).sort_values("strike_usd").reset_index(drop=True)
        if observation_rows
        else pd.DataFrame()
    )

    # Knots table
    knot_strikes = [
        math.exp(math.log(context.forward) + x * context.scale) for x in KNOTS
    ]
    knot_rows = []
    for i, (x, strike) in enumerate(zip(KNOTS, knot_strikes)):
        ref_k_iv = reference_vol(view, strike)
        knot_rows.append(
            {
                "knot_index": i + 1,
                "x_std_dev": x,
                "strike_usd": round(strike, 2),
                "filtered_iv": view.knot_ivs[i],
                "reference_iv": ref_k_iv,
            }
        )
    knots_df = pd.DataFrame.from_records(knot_rows)

    return {
        "view": view,
        "expiry_timestamp_ms": expiry_ms,
        "expiry_str": snapshot.get("expiry_str", ""),
        "spot_price": spot,
        "forward": context.forward,
        "forward_source": snapshot.get("forward_source"),
        "exercise_time": context.exercise_time,
        "time_to_close_hours": hours_left,
        "atm_vol": context.atm_vol,
        "butterfly_report": butterfly_report(view),
        "observations": observations_df,
        "knots": knots_df,
        "coefficients": pd.DataFrame.from_records(
            [
                {"segment": i + 1, "c0": row[0], "c1": row[1], "c2": row[2]}
                for i, row in enumerate(view.segment_coefficients)
            ]
        ),
    }, None


def make_spx_smile_figure(analysis: dict | None, error: str | None = None):
    """Generate high-clarity Plotly smile figure plotted over strike USD."""
    figure = go.Figure()

    if analysis is None or "view" not in analysis:
        figure.add_annotation(
            text=error or "Awaiting SPX 0DTE market quotes...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16, "color": "gray"},
        )
        figure.update_layout(
            template="plotly_dark",
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=500,
        )
        return figure

    view: SmileSnapshot = analysis["view"]
    context = view.context
    spot = analysis["spot_price"]

    # Determine plot strike range: from lowest observation to highest observation
    strikes_obs = np.asarray([o.strike for o in view.observations])
    if len(strikes_obs):
        k_min = max(spot * 0.95, float(strikes_obs.min()) - 10)
        k_max = min(spot * 1.05, float(strikes_obs.max()) + 10)
    else:
        k_min, k_max = spot - 80, spot + 80

    k_dense = np.linspace(k_min, k_max, 250)
    x_dense = context.coordinates(k_dense)

    # 1. Reference Total Variance Cubic Smile (dashed line)
    ref_vols = [reference_vol(view, k) for k in k_dense]
    if all(v is not None for v in ref_vols):
        ref_vols = [v * 100 for v in ref_vols]
        figure.add_trace(
            go.Scatter(
                x=k_dense,
                y=ref_vols,
                mode="lines",
                line={"color": "#FF7F50", "width": 2, "dash": "dash"},
                name="Total Variance Smile (Roger Lee wings)",
                hovertemplate="Strike: %{x:,.1f}<br>Reference IV: %{y:.2f}%<extra></extra>",
            )
        )

    # 2. Kalman Filtered Smile (Bayesian posterior natural cubic spline)
    # Clip evaluation to [-3, 3] std devs for display
    inside_k = (x_dense >= -3.0) & (x_dense <= 3.0)
    if inside_k.any():
        filtered_vols = view.volatility(x_dense[inside_k], display=True) * 100
        figure.add_trace(
            go.Scatter(
                x=k_dense[inside_k],
                y=filtered_vols,
                mode="lines",
                line={"color": "#00E5FF", "width": 3},
                name="Kalman Filtered Smile (Bayesian update)",
                hovertemplate="Strike: %{x:,.1f}<br>Filtered IV: %{y:.2f}%<extra></extra>",
            )
        )

    # 3. Observed Mid IVs (markers with bid/ask spread error bars)
    obs_df = analysis["observations"]
    if not obs_df.empty:
        iv_pct = obs_df["observed_mid_iv"] * 100
        spread_pct = (obs_df["iv_spread"] * 100) / 2.0
        figure.add_trace(
            go.Scatter(
                x=obs_df["strike_usd"],
                y=iv_pct,
                mode="markers",
                marker={"color": "#76FF03", "size": 7, "symbol": "circle"},
                error_y={"type": "data", "array": spread_pct, "visible": True, "color": "rgba(118, 255, 3, 0.4)"},
                name="Observed Mid IV (± Spread/2)",
                hovertemplate="Strike: %{x:,.1f} (%{customdata})<br>Observed Mid IV: %{y:.2f}%<extra></extra>",
                customdata=obs_df["right"],
            )
        )

    # 4. Filtered Knots (diamond markers at 9 standardized knots)
    knot_strikes = [math.exp(math.log(context.forward) + x * context.scale) for x in KNOTS]
    knot_vols = np.asarray(view.knot_ivs) * 100
    figure.add_trace(
        go.Scatter(
            x=knot_strikes,
            y=knot_vols,
            mode="markers",
            marker={"color": "#E040FB", "size": 9, "symbol": "diamond"},
            name="Kalman State Knots (9 knots)",
            hovertemplate="Knot: %{x:,.1f}<br>Knot IV: %{y:.2f}%<extra></extra>",
        )
    )

    # 5. ATM Spot Reference Line
    figure.add_vline(
        x=spot,
        line_width=1.5,
        line_dash="dot",
        line_color="#FFFFFF",
        annotation_text=f"ATM Spot: {spot:,.1f}",
        annotation_position="top left",
    )

    figure.update_layout(
        title={
            "text": f"<b>SPX 0DTE Volatility Smile Calibration</b> — Expiry {analysis.get('expiry_str', '')} ({analysis.get('time_to_close_hours', 0.0):.2f} hrs to close)",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={
            "title": "Strike Price (USD)",
            "tickformat": ",.0f",
            "gridcolor": "#333333",
        },
        yaxis={
            "title": "Implied Volatility (%)",
            "ticksuffix": "%",
            "gridcolor": "#333333",
        },
        template="plotly_dark",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        height=540,
        margin={"l": 60, "r": 40, "t": 90, "b": 60},
    )

    return figure

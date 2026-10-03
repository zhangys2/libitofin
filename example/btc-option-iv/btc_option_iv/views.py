"""Pure notebook views: no feed ownership, filter updates, or mutable UI state."""

import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from .kalman import KNOTS
from .reference import butterfly_report, reference_vol

MAX_QUOTE_AGE_SECONDS = 10.0
QUOTE_TABLE_LIMIT = 100
QUOTE_COLUMNS = [
    "instrument",
    "type",
    "strike_usd",
    "expiry_days",
    "bid_btc",
    "ask_btc",
    "bid_amount",
    "ask_amount",
    "index_usd",
    "forward_usd",
    "bid_iv",
    "mid_iv",
    "ask_iv",
    "age_seconds",
]


def make_surface_figure(
    rows, *, now_ms=None, max_age_seconds=MAX_QUOTE_AGE_SECONDS, grid_size=36
):
    """The original unfiltered, fresh mid-IV surface; convex-hull gaps stay null."""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    buckets = defaultdict(list)
    for row in rows.values():
        if row.mid_iv is None or now - row.timestamp_ms > max_age_seconds * 1000:
            continue
        if row.expiry <= 0 or row.forward <= 0 or row.strike <= 0:
            continue
        days, log_moneyness = row.expiry * 365, math.log(row.strike / row.forward)
        if (
            math.isfinite(days)
            and math.isfinite(log_moneyness)
            and math.isfinite(row.mid_iv)
        ):
            buckets[(round(days, 3), round(log_moneyness, 5))].append(row.mid_iv)
    points = [(x, y, float(np.median(values))) for (x, y), values in buckets.items()]
    figure = go.Figure()
    if len(points) < 4:
        if points:
            x, y, z = zip(*points)
            figure.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": z,
                        "colorscale": "Turbo",
                        "colorbar": {"title": "IV (decimal)"},
                    },
                    name="observed mid IV",
                )
            )
        else:
            figure.add_annotation(
                text="Waiting for fresh two-sided option quotes…", showarrow=False
            )
    else:
        x_values, y_values, z_values = np.asarray(points, dtype=float).T
        grid_z = None
        if np.ptp(x_values) > 0 and np.ptp(y_values) > 0:
            grid_x, grid_y = np.meshgrid(
                np.linspace(x_values.min(), x_values.max(), grid_size),
                np.linspace(y_values.min(), y_values.max(), grid_size),
            )
            try:
                grid_z = griddata(
                    (x_values, y_values), z_values, (grid_x, grid_y), method="linear"
                )
            except QhullError:
                pass
        if grid_z is not None and np.isfinite(grid_z).sum() >= 4:
            safe_grid_z = [
                [float(v) if np.isfinite(v) else None for v in row] for row in grid_z
            ]
            figure.add_trace(
                go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=safe_grid_z,
                    connectgaps=False,
                    colorscale="Turbo",
                    colorbar={"title": "IV (decimal)"},
                    name="interpolated mid IV",
                    hovertemplate="tenor=%{x:.1f}d<br>ln(K/F)=%{y:.3f}<br>IV=%{z:.3f}<extra></extra>",
                )
            )
        else:
            figure.add_trace(
                go.Scatter3d(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    mode="markers",
                    name="observed mid IV",
                )
            )
    figure.update_layout(
        title=f"Deribit BTC inverse options — fresh mid IVs ({len(points)} points)",
        scene={
            "xaxis_title": "Days to expiry",
            "yaxis_title": "Log-moneyness ln(K/F)",
            "zaxis_title": "Implied volatility (decimal)",
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        uirevision="btc-live-vol-surface",
    )
    return figure


def make_quotes_dataframe(snapshot: dict):
    """Show the 100 most recently updated option quotes and their solved IVs."""
    now = int(time.time() * 1000)
    records = []
    for name, quote in snapshot["latest_options"].items():
        instrument = snapshot["instruments"].get(name)
        if instrument is None or (quote.bid_price is None and quote.ask_price is None):
            continue
        forwards = snapshot["book"].forward_data(instrument)
        iv_row = snapshot["rows"].get(name)
        forward = quote.underlying_price
        if forward is None:
            forward = iv_row.forward if iv_row is not None else forwards.forward
        index = quote.index_price if quote.index_price is not None else forwards.index
        records.append(
            {
                "quote_timestamp_ms": quote.timestamp_ms,
                "instrument": name,
                "type": instrument.option_type.upper(),
                "strike_usd": instrument.strike,
                "expiry_days": max(
                    0.0,
                    (instrument.expiration_timestamp_ms - quote.timestamp_ms)
                    / 86_400_000,
                ),
                "bid_btc": quote.bid_price,
                "ask_btc": quote.ask_price,
                "bid_amount": quote.bid_amount,
                "ask_amount": quote.ask_amount,
                "index_usd": index,
                "forward_usd": forward,
                "bid_iv": iv_row.bid_iv if iv_row is not None else None,
                "mid_iv": iv_row.mid_iv if iv_row is not None else None,
                "ask_iv": iv_row.ask_iv if iv_row is not None else None,
                "age_seconds": max(0.0, (now - quote.timestamp_ms) / 1000),
            }
        )
    if not records:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["quote_timestamp_ms", "instrument"], ascending=[False, True])
        .head(QUOTE_TABLE_LIMIT)
        .loc[:, QUOTE_COLUMNS]
        .reset_index(drop=True)
    )


def make_smile_reader(snapshot: dict, expiry_timestamp_ms: int | None, *, now_ms=None):
    """Read an immutable filter snapshot; NEVER reconstruct or update a filter."""
    if expiry_timestamp_ms is None:
        return None, "Select an expiry after fresh option quotes arrive."
    view = snapshot.get("kalman_views", {}).get(expiry_timestamp_ms)
    if view is None:
        return None, "Waiting for event-time filter data for this expiry."
    if view.knot_ivs is None:
        return None, "; ".join(
            view.warnings
        ) or "Waiting for two fresh, distinct in-range strikes."
    now = int(time.time() * 1000) if now_ms is None else now_ms
    context = view.context
    coefficient_rows = [
        {
            "segment": i,
            "x_start_std_dev": KNOTS[i],
            "x_end_std_dev": KNOTS[i + 1],
            "mid_iv_at_start": view.knot_ivs[i],
            "a": a,
            "b": b,
            "c": c,
        }
        for i, (a, b, c) in enumerate(view.segment_coefficients)
    ]
    observation_rows = []
    for o, diagnostic in zip(view.observations, view.diagnostics):
        x = float(context.coordinates([o.strike])[0])
        in_range = -3 <= x <= 3
        filtered = float(view.volatility([x], display=False)[0]) if in_range else None
        reference = reference_vol(view, o.strike) if in_range else None
        observation_rows.append(
            {
                "instrument": o.instrument,
                "premium_version": o.version,
                "strike_usd": o.strike,
                "x_std_dev": x,
                "observed_mid_iv": o.mid_iv,
                "iv_spread": o.spread,
                "fitted_minus_observed": reference - o.mid_iv
                if reference is not None
                else None,
                "filtered_minus_observed": filtered - o.mid_iv
                if filtered is not None
                else None,
                "innovation": diagnostic.innovation,
                "innovation_z": diagnostic.innovation_z,
                "measurement_status": diagnostic.status,
                "used_in_fit": in_range,
                "quotes_combined": o.alternatives,
                "age_seconds": max(0.0, (now - o.timestamp_ms) / 1000),
            }
        )
    live = snapshot.get("feed_running", False)
    age = view.last_accepted_age
    if age is not None:
        age += max(0.0, (now - snapshot.get("published_ms", now)) / 1000)
    status = (
        "paused (frozen view)"
        if not live
        else ("stale" if age is not None and age >= 10 else view.status)
    )
    ref_sec = view.reference_section
    return {
        "view": view,
        "expiry_timestamp_ms": expiry_timestamp_ms,
        "forward": context.forward,
        "exercise_time": context.exercise_time,
        "atm_vol": context.atm_vol,
        "smoothing": ref_sec.smoothing if ref_sec is not None else 0.01,
        "butterfly_report": butterfly_report(view),
        "reference_section": ref_sec,
        "filter_status": status,
        "last_accepted_age": age,
        "coefficients": pd.DataFrame.from_records(coefficient_rows),
        "observations": pd.DataFrame.from_records(observation_rows),
        "knots": pd.DataFrame(
            {"x_std_dev": KNOTS, "filtered_knot_iv_unclipped": view.knot_ivs}
        ),
    }, None


def make_smile_figure(analysis: dict | None, error: str | None = None):
    figure = go.Figure()
    if analysis is None:
        figure.add_annotation(text=error or "No smile fit available.", showarrow=False)
    else:
        view = analysis["view"]
        x_observed = view.context.coordinates([o.strike for o in view.observations])
        inside = (x_observed >= -3) & (x_observed <= 3)
        support = (
            view.context.coordinates(view.observed_support)
            if view.observed_support
            else x_observed
        )
        plot_min = max(-3.0, float(support.min())) if len(support) else 1.0
        plot_max = min(3.0, float(support.max())) if len(support) else -1.0
        if plot_min <= plot_max:
            x = np.linspace(plot_min, plot_max, 200)
            ref_strikes = [
                math.exp(math.log(view.context.forward) + xi * view.context.scale)
                for xi in x
            ]
            ref_y = [reference_vol(view, k) for k in ref_strikes]
            if all(v is not None for v in ref_y):
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=ref_y,
                        mode="lines",
                        line={"dash": "dash"},
                        name="current regularized fit",
                    )
                )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=view.volatility(x),
                    mode="lines",
                    name="Kalman filtered smile",
                )
            )
            y_observed = np.asarray([o.mid_iv for o in view.observations])
            if inside.any():
                figure.add_trace(
                    go.Scatter(
                        x=x_observed[inside],
                        y=y_observed[inside],
                        mode="markers",
                        name="observed mid IV",
                    )
                )
            knots = np.asarray(KNOTS)
            plotted = (knots >= plot_min) & (knots <= plot_max)
            if plotted.any():
                figure.add_trace(
                    go.Scatter(
                        x=knots[plotted],
                        y=view.volatility(knots[plotted]),
                        mode="markers",
                        marker={"symbol": "x", "size": 9},
                        name="filtered knot IVs",
                    )
                )
        else:
            figure.add_annotation(
                text="No fresh observed support inside the knot range.", showarrow=False
            )
    figure.update_layout(
        title="Selected-expiry mid-IV smile — raw fit vs Kalman (visualization only)",
        xaxis_title="Standard-deviation log-moneyness",
        yaxis_title="Implied volatility (decimal)",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        # Keep zoom through quote refreshes, but reset it for a new expiry.
        uirevision=f"btc-expiry-smile-{analysis['expiry_timestamp_ms']}"
        if analysis
        else "btc-expiry-smile",
    )
    return figure

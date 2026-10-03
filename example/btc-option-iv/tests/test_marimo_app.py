"""Marimo wiring and pure snapshot views; never connect to the public feed."""

import ast
import copy
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_APP = Path(__file__).resolve().parents[1] / "live_vol_surface.py"


def test_live_surface_app_is_python_and_syntax_valid():
    source = _APP.read_text(encoding="utf-8")
    compile(source, "live_vol_surface.py", "exec")
    assert '__generated_with = "0.25.0"' in source
    assert 'app = marimo.App(width="full")' in source


def test_live_quotes_surface_and_filter_controls_are_reactive_outputs():
    source = _APP.read_text(encoding="utf-8")
    assert "mo.state(" in source
    assert "mo.ui.plotly(surface_figure)" in source
    assert "make_quotes_dataframe(market_snapshot)" in source
    assert 'label="Connect to live Deribit quotes"' in source
    assert "await feed_controller.set_enabled(live_enabled.value)" in source
    assert "Expiry for cubic smile reader" in source
    assert "process_iv_rate=process_rate.value / 100.0" in source
    assert "measurement_sd=measurement_sd.value / 100.0" in source
    assert "feed_controller.reset_filters()" in source
    assert "filters.ingest" not in source  # UI never assimilates a quote.


def test_all_widget_values_are_read_outside_their_creator_cells():
    cells = _APP.read_text(encoding="utf-8").split("@app.cell")
    creator = next(
        cell for cell in cells if 'label="Expiry for cubic smile reader"' in cell
    )
    assert "expiry_selector.value" not in creator
    controls = next(cell for cell in cells if "process_rate = mo.ui.number" in cell)
    for name in ("process_rate", "measurement_sd", "reset_filters"):
        assert f"{name}.value" not in controls
    assert any("selected_expiry_ms = expiry_selector.value" in cell for cell in cells)
    # Reset has its own cell: changing noise controls cannot repeat a button click.
    reset_cell = next(
        cell for cell in cells if "feed_controller.reset_filters()" in cell
    )
    assert "process_rate" not in reset_cell and "measurement_sd" not in reset_cell


def _execute_cell(cell, namespace):
    cell = copy.deepcopy(cell)
    cell.name = "test_cell"
    cell.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
    scope = {}
    exec(compile(module, str(_APP), "exec"), scope)  # noqa: S102 -- trusted local notebook cells only
    result = scope["test_cell"](
        **{arg.arg: namespace[arg.arg] for arg in cell.args.args}
    )
    if result is not None:
        returned = cell.body[-1].value
        names = returned.elts if isinstance(returned, ast.Tuple) else [returned]
        namespace.update({name.id: value for name, value in zip(names, result)})


def _selector_render_loop(snapshot):
    import marimo as mo
    cells = [
        n for n in ast.parse(_APP.read_text()).body if isinstance(n, ast.FunctionDef)
    ]
    namespace = {"mo": mo, "time": time, "market_snapshot": snapshot}
    # New selector-only states, if present. Running the old notebook remains
    # possible, so the regression proves the selection loss rather than layout.
    for cell in cells:
        if "mo.state" in ast.unparse(cell) and "get_market_snapshot" not in ast.unparse(
            cell
        ):
            _execute_cell(cell, namespace)
    creator = next(
        cell for cell in cells if "Expiry for cubic smile reader" in ast.unparse(cell)
    )
    bridge = next(
        (
            cell
            for cell in cells
            if "set_expiry_catalog" in [a.arg for a in cell.args.args]
        ),
        None,
    )

    def render(snapshot):
        namespace["market_snapshot"] = snapshot
        if bridge is not None:
            _execute_cell(bridge, namespace)
        _execute_cell(creator, namespace)
        return namespace["expiry_selector"]

    return render, creator


def test_selected_expiry_survives_quote_refresh_and_catalog_changes():
    first, second, third = [
        1_800_000_000_000 + days * 86_400_000 for days in (1, 7, 30)
    ]

    def snapshot(expiries, frame=0):
        return {
            "instruments": {
                str(e): SimpleNamespace(expiration_timestamp_ms=e) for e in expiries
            },
            "frame": frame,
        }

    render, creator = _selector_render_loop(snapshot([first, second]))
    selector = render(snapshot([first, second]))
    selected_key = next(
        key for key, value in selector.options.items() if value == second
    )
    selector._update([selected_key])  # Same conversion/on_change as a browser click.
    assert selector.value == second
    for frame in range(3):
        assert render(snapshot([first, second], frame)).value == second
    assert render(snapshot([first, second, third])).value == second
    assert render(snapshot([first, third])).value == first
    assert "market_snapshot" not in [arg.arg for arg in creator.args.args]


def _modules():
    import numpy as np
    import pandas as pd
    import scipy
    import plotly
    from btc_option_iv.kalman import FilterManager
    from btc_option_iv.views import make_smile_figure, make_smile_reader

    return FilterManager, make_smile_reader, make_smile_figure


def _smile_snapshot(*, points=None):
    FilterManager, _, _ = _modules()
    now_ms = 1_800_000_000_000
    expiry_ms = now_ms + 30 * 86_400_000
    expiry = 30 / 365
    forward, atm = 100_000.0, 0.30
    scale = atm * math.sqrt(expiry)
    points = (
        points
        if points is not None
        else [-3.5] + [i / 4 for i in range(-12, 13)] + [3.5]
    )
    instruments, rows = {}, {}
    for i, x in enumerate(points):
        name = f"option-{i}"
        instruments[name] = SimpleNamespace(
            expiration_timestamp_ms=expiry_ms, option_type="c"
        )
        iv = atm + 0.01 * abs(x)
        rows[name] = SimpleNamespace(
            instrument_name=name,
            timestamp_ms=now_ms,
            strike=forward * math.exp(x * scale),
            mid_iv=iv,
            bid_iv=iv - 0.005,
            ask_iv=iv + 0.005,
            bid_premium_btc=iv - 0.005,
            ask_premium_btc=iv + 0.005,
            forward=forward,
            expiry=expiry,
        )
    if 0 in points:
        instruments["atm-put"] = SimpleNamespace(
            expiration_timestamp_ms=expiry_ms, option_type="p"
        )
        rows["atm-put"] = SimpleNamespace(
            instrument_name="atm-put",
            timestamp_ms=now_ms,
            strike=forward,
            mid_iv=atm,
            bid_iv=atm - 0.005,
            ask_iv=atm + 0.005,
            bid_premium_btc=atm - 0.005,
            ask_premium_btc=atm + 0.005,
            forward=forward,
            expiry=expiry,
        )
    manager = FilterManager()
    for name, row in rows.items():
        manager.ingest(row, instruments[name], now_ms=now_ms, now=0)
    snapshot = {
        "instruments": instruments,
        "rows": rows,
        "kalman_views": manager.snapshots(now=0),
        "published_ms": now_ms,
        "feed_running": True,
    }
    return snapshot, expiry_ms, now_ms, manager


def test_reader_has_nine_knots_and_both_curves_with_correct_residuals():
    from itofin.termstructures import TotalVarianceCubicSmileSection

    _, reader, plot = _modules()
    snapshot, expiry_ms, now_ms, manager = _smile_snapshot()
    before = manager.snapshots(now=0)
    analysis, error = reader(snapshot, expiry_ms, now_ms=now_ms)
    assert error is None
    view = analysis["view"]
    assert len(view.knot_ivs) == 9
    assert len(analysis["coefficients"]) == 8
    assert len(analysis["knots"]) == 9
    table = analysis["observations"]
    assert len(table) == 27
    assert not table.iloc[0]["used_in_fit"] and not table.iloc[-1]["used_in_fit"]
    assert table.iloc[1:-1]["used_in_fit"].all()
    assert table.iloc[0:1]["fitted_minus_observed"].isna().all()
    assert table.iloc[-1:]["filtered_minus_observed"].isna().all()
    assert table[table["strike_usd"] == 100_000.0].iloc[0]["quotes_combined"] == 2
    reference = TotalVarianceCubicSmileSection(
        [o.strike for o in view.observations],
        [o.mid_iv for o in view.observations],
        view.context.forward,
        view.context.exercise_time,
        view.context.atm_vol,
    )
    for record in table.to_dict("records"):
        if record["used_in_fit"]:
            assert record["fitted_minus_observed"] == pytest.approx(
                reference.volatility(record["strike_usd"])
                - record["observed_mid_iv"],
                abs=1e-12,
            )
            assert record["filtered_minus_observed"] == pytest.approx(
                view.volatility([record["x_std_dev"]], display=False)[0]
                - record["observed_mid_iv"],
                abs=1e-12,
            )
    figure = plot(analysis)
    assert [trace.name for trace in figure.data] == [
        "current regularized fit",
        "Kalman filtered smile",
        "observed mid IV",
        "filtered knot IVs",
    ]
    assert min(figure.data[0].x) >= -3 and max(figure.data[0].x) <= 3
    assert manager.snapshots(now=0) == before  # Reading/plotting is pure.


def test_selected_expiry_routes_to_its_view_and_resets_only_its_plot_zoom():
    from dataclasses import replace

    _, reader, plot = _modules()
    snapshot, first, now_ms, _ = _smile_snapshot(points=[-0.5, 0, 0.5])
    second = first + 30 * 86_400_000
    original = snapshot["kalman_views"][first]
    other = replace(
        original,
        context=replace(original.context, exercise_time=60 / 365),
        knot_ivs=tuple(iv + 0.1 for iv in original.knot_ivs),
    )
    snapshot["kalman_views"][second] = other
    a, error = reader(snapshot, first, now_ms=now_ms)
    assert error is None and a["view"] is original
    b, error = reader(snapshot, second, now_ms=now_ms)
    assert error is None and b["view"] is other
    assert b["expiry_timestamp_ms"] == second
    assert b["exercise_time"] == 60 / 365
    assert (
        b["knots"]["filtered_knot_iv_unclipped"].mean()
        > a["knots"]["filtered_knot_iv_unclipped"].mean()
    )
    assert plot(a).layout.uirevision != plot(b).layout.uirevision
    again, _ = reader(snapshot, second, now_ms=now_ms + 3000)
    assert plot(again).layout.uirevision == plot(b).layout.uirevision


def test_sparse_reader_keeps_nine_knots_and_plots_only_observed_support():
    _, reader, plot = _modules()
    snapshot, expiry_ms, now_ms, _ = _smile_snapshot(points=[-0.5, -0.25, 0, 0.25, 0.5])
    analysis, error = reader(snapshot, expiry_ms, now_ms=now_ms)
    assert error is None
    assert len(analysis["observations"]) == 5
    observations = analysis["observations"]["x_std_dev"]
    for trace in plot(analysis).data:
        assert min(trace.x) >= min(observations)
        assert max(trace.x) <= max(observations)


def test_reader_waits_for_two_distinct_in_range_quotes():
    _, reader, plot = _modules()
    snapshot, expiry_ms, now_ms, _ = _smile_snapshot(points=[0, 4])
    analysis, error = reader(snapshot, expiry_ms, now_ms=now_ms)
    assert analysis is None
    assert "at least 2 in-range observations" in error
    assert len(plot(analysis, error).data) == 0


def test_empty_quote_table_and_surface_remain_available():
    _modules()
    from btc_option_iv.views import (
        QUOTE_COLUMNS,
        make_quotes_dataframe,
        make_surface_figure,
    )

    table = make_quotes_dataframe(
        {"latest_options": {}, "rows": {}, "instruments": {}, "book": None}
    )
    assert list(table.columns) == QUOTE_COLUMNS
    assert len(make_surface_figure({}).data) == 0

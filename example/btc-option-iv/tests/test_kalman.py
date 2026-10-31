"""Offline contracts for the nine-knot temporal smile prototype."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import numpy as np
import scipy

from btc_option_iv.kalman import (
    KNOTS,
    FilterConfig,
    FilterManager,
    SmileContext,
    basis_matrix,
    kalman_update,
    process_covariance,
    transport,
)
from btc_option_iv.kalman_regime import RegimeConfig, RegimeDetector

NOW_MS = 1_800_000_000_000
EXPIRY_MS = NOW_MS + 30 * 86_400_000
CONTEXT = SmileContext(100_000.0, 30 / 365, 0.3)


def quote(
    x,
    iv=0.3,
    width=0.01,
    *,
    timestamp=NOW_MS,
    name=None,
    expiry=EXPIRY_MS,
    option_type="c",
):
    name = name or f"q-{x}-{option_type}"
    strike = math.exp(math.log(CONTEXT.forward) + x * CONTEXT.scale)
    row = SimpleNamespace(
        instrument_name=name,
        option_type=option_type,
        timestamp_ms=timestamp,
        strike=strike,
        mid_iv=iv,
        bid_iv=iv - width / 2,
        ask_iv=iv + width / 2,
        bid_premium_btc=iv - width / 2,
        ask_premium_btc=iv + width / 2,
        forward=CONTEXT.forward,
        expiry=CONTEXT.exercise_time,
    )
    instrument = SimpleNamespace(
        expiration_timestamp_ms=expiry, option_type=option_type
    )
    return row, instrument


def ingest(manager, x, iv=0.3, width=0.01, *, now=0.0, **kwargs):
    timestamp = NOW_MS + int(now * 1000)
    row, instrument = quote(x, iv, width, timestamp=timestamp, **kwargs)
    manager.ingest(row, instrument, now_ms=timestamp, now=now)
    return row, instrument


def warm(manager, *, now=0.0, expiry=EXPIRY_MS):
    for x in [-1.4, -1.0, -0.6, 0.0, 0.6, 1.0, 1.4]:
        ingest(manager, x, now=now, expiry=expiry)
    return manager.snapshots(now=now)[expiry]


def test_scalar_update_matches_closed_form_and_joseph_covariance():
    mean = np.zeros(9)
    covariance = np.eye(9) * 2
    h = np.eye(9)[[4]]
    result, p, innovation, z = kalman_update(
        mean, covariance, h, np.array([3.0]), np.array([1.0])
    )
    assert result[4] == pytest.approx(2.0)
    assert p[4, 4] == pytest.approx(2 / 3)
    assert innovation[0] == 3
    assert z[0] == pytest.approx(math.sqrt(3))
    assert np.allclose(p, p.T)
    assert np.linalg.eigvalsh(p).min() >= 0


def test_batch_update_matches_independent_linear_solve():
    rng = np.random.default_rng(29)
    mean = rng.normal(size=9)
    a = rng.normal(size=(9, 9))
    p = a @ a.T + np.eye(9)
    h = basis_matrix([-1.1, 0.0, 0.7])
    y, r = np.array([0.25, 0.3, 0.35]), np.array([0.001, 0.002, 0.003])
    gain = np.linalg.solve(h @ p @ h.T + np.diag(r), h @ p).T
    expected = mean + gain @ (y - h @ mean)
    updated, covariance, _, _ = kalman_update(mean, p, h, y, r)
    assert updated == pytest.approx(expected)
    assert covariance == pytest.approx(p - gain @ h @ p, abs=1e-12)


def test_basis_and_local_coefficients_match_rust_natural_cubic():
    from itofin.termstructures import CubicSmileSection
    from scipy.interpolate import CubicSpline

    values = 0.3 + 0.02 * np.asarray(KNOTS) ** 2
    strikes = [math.exp(math.log(CONTEXT.forward) + x * CONTEXT.scale) for x in KNOTS]
    core = CubicSmileSection(
        strikes,
        values,
        CONTEXT.forward,
        CONTEXT.exercise_time,
        CONTEXT.atm_vol,
        smoothing=0,
    )
    spline = CubicSpline(KNOTS, values, bc_type="natural")
    for x in np.linspace(-3, 3, 31):
        assert (basis_matrix([x]) @ values).item() == pytest.approx(
            core.volatility_at_std_dev(x), abs=1e-12
        )
    assert np.column_stack((spline.c[2], spline.c[1], spline.c[0])) == pytest.approx(
        np.asarray(core.segment_coefficients)
    )


def test_inverse_spread_precision_floor_and_process_units():
    config = FilterConfig()
    assert config.measurement_variance(0.01) == pytest.approx(0.005**2)
    assert config.measurement_variance(0.02) == pytest.approx(2 * 0.005**2)
    assert config.measurement_variance(0.0) == config.measurement_variance(0.001)
    q = process_covariance(config, 3)
    assert np.diag(q) == pytest.approx(np.full(9, 3e-6))
    assert q[0, 1] == pytest.approx(3e-6 * 0.9 * math.exp(-1.5))
    assert np.linalg.eigvalsh(q).min() > 0
    for bad in [-1, math.nan, math.inf]:
        with pytest.raises(ValueError):
            FilterConfig(process_iv_rate=bad)
        with pytest.raises(ValueError):
            config.measurement_variance(bad)
    with pytest.raises(ValueError):
        FilterConfig(measurement_sd=0)


def test_transport_identity_and_strike_space_linear_curve():
    values = 0.3 + 0.02 * np.asarray(KNOTS)
    p = np.eye(9) * 0.001
    same, same_p, j = transport(values, p, CONTEXT, CONTEXT)
    assert same == pytest.approx(values)
    assert same_p == pytest.approx(p)
    assert j == pytest.approx(np.eye(9))
    new = replace(CONTEXT, forward=CONTEXT.forward * math.exp(0.1 * CONTEXT.scale))
    projected, projected_p, j = transport(values, p, CONTEXT, new)
    assert projected == pytest.approx(0.3 + 0.02 * (np.asarray(KNOTS) + 0.1))
    extra = projected_p - j @ p @ j.T
    assert extra[-1, -1] == pytest.approx((0.01 * 0.1 / 0.25) ** 2)
    assert np.diag(extra)[:-1] == pytest.approx(np.zeros(8), abs=1e-12)
    with pytest.raises(ValueError, match="overhang"):
        transport(
            values,
            p,
            CONTEXT,
            replace(new, forward=CONTEXT.forward * math.exp(0.3 * CONTEXT.scale)),
        )


def test_slope_clamp_jacobian_and_extension_sd():
    values = 0.5 + 0.2 * np.asarray(KNOTS)
    new = replace(CONTEXT, forward=CONTEXT.forward * math.exp(0.25 * CONTEXT.scale))
    projected, p, j = transport(values, np.eye(9), CONTEXT, new)
    assert projected[-1] == pytest.approx(values[-1] + 0.1 * 0.25)
    assert j[-1] == pytest.approx(np.eye(9)[-1])
    assert (p - j @ j.T)[-1, -1] == pytest.approx(0.01**2)
    delta = 1e-6
    for k in range(9):
        bumped = values.copy()
        bumped[k] += delta
        result, _, _ = transport(bumped, np.eye(9), CONTEXT, new)
        assert (result[-1] - projected[-1]) / delta == pytest.approx(j[-1, k], abs=1e-7)


def test_sparse_bootstrap_single_quote_update_and_duplicate_heartbeat():
    manager = FilterManager()
    ingest(manager, -0.6)
    assert manager.snapshots(now=0)[EXPIRY_MS].knot_ivs is None
    ingest(manager, 0.6)
    initial = manager.snapshots(now=0)[EXPIRY_MS]
    assert len(initial.knot_ivs) == 9
    assert len(initial.segment_coefficients) == 8
    assert np.diag(initial.covariance) == pytest.approx(np.full(9, 0.05**2))
    row, instrument = ingest(manager, 0.6, 0.301, now=1)
    updated = manager.snapshots(now=1)[EXPIRY_MS]
    assert updated.updates == 1
    manager.ingest(row, instrument, now_ms=NOW_MS + 1000, now=1)
    assert manager.snapshots(now=1)[EXPIRY_MS].covariance == updated.covariance
    # Timestamp-only heartbeat does not become another measurement.
    heartbeat = SimpleNamespace(**{**vars(row), "timestamp_ms": NOW_MS + 2000})
    manager.ingest(heartbeat, instrument, now_ms=NOW_MS + 2000, now=2)
    assert manager.snapshots(now=2)[EXPIRY_MS].updates == 1
    assert manager.snapshots(now=2)[EXPIRY_MS].last_accepted_age == 1
    assert updated.knot_ivs == manager.snapshots(now=1)[EXPIRY_MS].knot_ivs


def test_reference_revaluation_and_old_versions_do_not_assimilate():
    manager = FilterManager()
    warm(manager)
    before = manager.snapshots(now=0)[EXPIRY_MS]
    row, instrument = quote(0.6)
    row.mid_iv, row.bid_iv, row.ask_iv = 0.301, 0.296, 0.306
    manager.ingest(row, instrument, now_ms=NOW_MS, now=0)
    assert manager.snapshots(now=0)[EXPIRY_MS].updates == before.updates
    ingest(manager, 0.6, 0.302, now=2)
    newest = manager.snapshots(now=2)[EXPIRY_MS]
    manager.ingest(*quote(0.6, 0.1), now_ms=NOW_MS + 2000, now=2)
    assert manager.snapshots(now=2)[EXPIRY_MS].knot_ivs == newest.knot_ivs


def test_quote_selection_uses_iv_spread_and_no_replay_on_switch():
    manager = FilterManager()
    ingest(manager, -0.6)
    ingest(manager, 0.6, width=0.02, option_type="p")
    ingest(manager, 0.6, width=0.01, option_type="c", now=1)
    view = manager.snapshots(now=1)[EXPIRY_MS]
    assert view.observations[-1].option_type == "c"
    updates = view.updates
    # Revalue the put spread without changing its premium: selecting an already
    # bootstrapped version must not count it a second time.
    row, instrument = quote(0.6, width=0.02, option_type="p", timestamp=NOW_MS + 1000)
    row.bid_iv, row.ask_iv = 0.299, 0.301
    manager.ingest(row, instrument, now_ms=NOW_MS + 1000, now=1)
    assert manager.snapshots(now=1)[EXPIRY_MS].updates == updates


def test_independent_expiries_snapshots_controls_and_retirement():
    manager = FilterManager()
    warm(manager)
    other = EXPIRY_MS + 86_400_000
    warm(manager, expiry=other)
    snapshots = manager.snapshots(now=0)
    assert set(snapshots) == {EXPIRY_MS, other}
    original = snapshots[EXPIRY_MS]
    manager.configure(FilterConfig(process_iv_rate=0.002), now=1)
    assert manager.snapshots(now=1)[EXPIRY_MS].knot_ivs == original.knot_ivs
    manager.tick(now_ms=NOW_MS + 2000, now=2)
    assert (
        manager.snapshots(now=2)[EXPIRY_MS].covariance[0][0]
        >= original.covariance[0][0] + 5e-6 - 1e-12
    )
    assert snapshots[EXPIRY_MS].covariance == original.covariance
    manager.retire({other}, now_ms=NOW_MS)
    assert set(manager.snapshots(now=2)) == {other}
    manager.reset_all(now_ms=NOW_MS + 2000, now=2)
    reset_snap = manager.snapshots(now=2)[other]
    assert reset_snap.reset_reason == "manual reset"
    assert reset_snap.knot_ivs is not None
    assert reset_snap.status == "tracking"


def test_gaps_grow_uncertainty_mark_stale_and_require_fresh_bootstrap():
    manager = FilterManager()
    initial = warm(manager)
    manager.tick(now_ms=NOW_MS + 11_000, now=11)
    stale = manager.snapshots(now=11)[EXPIRY_MS]
    assert stale.status == "stale"
    assert stale.covariance[4][4] > initial.covariance[4][4]
    manager.tick(now_ms=NOW_MS + 61_000, now=61)
    assert manager.snapshots(now=61)[EXPIRY_MS].knot_ivs is None
    ingest(manager, -0.6, now=62)
    assert manager.snapshots(now=62)[EXPIRY_MS].knot_ivs is None
    ingest(manager, 0.6, now=62)
    assert manager.snapshots(now=62)[EXPIRY_MS].status == "tracking"


def test_isolated_atm_outlier_cannot_reset_normalization_before_gating():
    manager = FilterManager()
    before = warm(manager)
    ingest(manager, 0.0, iv=1.0, now=1)
    view = manager.snapshots(now=1)[EXPIRY_MS]
    assert view.updates == before.updates
    assert view.reset_reason == before.reset_reason
    assert max(abs(np.asarray(view.knot_ivs) - 0.3)) < 0.01
    assert any(d.status == "gated" for d in view.diagnostics)


def test_three_distinct_strikes_trigger_regime_reset_not_one_repeated_strike():
    manager = FilterManager()
    warm(manager)
    for i, x in enumerate([-1.0, 0.6, 1.0]):
        ingest(manager, x, iv=0.8, now=1 + i * 0.1)
    view = manager.snapshots(now=1.2)[EXPIRY_MS]
    assert view.reset_reason == "multi-strike regime change"
    assert view.resets == 1
    another = FilterManager()
    warm(another)
    for i in range(3):
        ingest(another, 0.6, iv=0.8 + i * 0.01, now=1 + i * 0.1)
    assert another.snapshots(now=1.2)[EXPIRY_MS].resets == 0


def test_solver_failure_keeps_predicted_state_and_reports_failure(monkeypatch):
    import btc_option_iv.kalman as module

    manager = FilterManager()
    before = warm(manager)

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("injected factorization failure")

    monkeypatch.setattr(module, "cho_factor", fail)
    ingest(manager, 0.6, iv=0.301, now=1)
    after = manager.snapshots(now=1)[EXPIRY_MS]
    assert after.updates == before.updates
    assert after.last_accepted_age == 1
    assert np.isfinite(after.covariance).all()
    assert after.covariance[4][4] >= before.covariance[4][4]
    assert any("Numerical update skipped" in warning for warning in after.warnings)
    assert any(d.status == "solver_failed" for d in after.diagnostics)


def test_display_flooring_does_not_modify_internal_state_or_covariance():
    manager = FilterManager()
    warm(manager)
    state = manager._filters[EXPIRY_MS]  # Inject a legal linear Gaussian state.
    state.mean = 0.05 - 0.04 * np.asarray(KNOTS)
    raw, covariance = state.mean.copy(), state.covariance.copy()
    view = manager.snapshots(now=0)[EXPIRY_MS]
    assert view.volatility([3])[0] == 0
    assert view.volatility([3], display=False)[0] < 0
    assert view.warnings
    assert state.mean == pytest.approx(raw)
    assert state.covariance == pytest.approx(covariance)


def test_invalid_spreads_and_older_quotes_do_not_poison_filter():
    manager = FilterManager()
    before = warm(manager)
    row, instrument = quote(0.6)
    row.ask_iv = row.bid_iv - 0.01
    manager.ingest(row, instrument, now_ms=NOW_MS, now=0)
    assert manager.snapshots(now=0)[EXPIRY_MS].updates == before.updates
    assert np.isfinite(manager.snapshots(now=0)[EXPIRY_MS].covariance).all()
    row, instrument = quote(0.6)
    row.mid_iv = math.nan
    manager.ingest(row, instrument, now_ms=NOW_MS, now=0)
    assert np.isfinite(manager.snapshots(now=0)[EXPIRY_MS].knot_ivs).all()


def test_replay_stationary_jitter_and_sustained_step_acceptance():
    """Seed 430, 7 quotes/sec, independent 0.005 IV SD, 0.01 IV spread.

    Fixed physical strikes and fixed F/T. Source ATM IV is still estimated from
    quotes by the real manager, so covariance/curve transport is exercised.
    Sample at 3s, discard 30s bootstrap, compare the SAME selected raw quotes.
    """
    rng = np.random.default_rng(430)
    manager = FilterManager()
    measurement_x = [-1.4, -1, -0.6, 0, 0.6, 1, 1.4]
    evaluation_strikes = [quote(x)[0].strike for x in [-1, 0, 1]]
    filtered, reference = [], []
    before_step = None
    response_seconds = None
    for second in range(210):
        truth = 0.3 if second < 180 else 0.35
        for i, x in enumerate(measurement_x):
            ingest(manager, x, iv=truth + rng.normal(0, 0.005), now=second + i * 0.001)
        now = second + 0.01
        manager.tick(now_ms=NOW_MS + int(now * 1000), now=now)
        view = manager.snapshots(now=now)[EXPIRY_MS]
        evaluation_x = view.context.coordinates(evaluation_strikes)
        values = view.volatility(evaluation_x, display=False)
        if second == 179:
            before_step = float(values.mean())
        if (
            second >= 180
            and response_seconds is None
            and float(values.mean()) >= before_step + 0.9 * (0.35 - before_step)
        ):
            response_seconds = second - 180 + 0.01
        if 30 <= second < 180 and second % 3 == 0:
            filtered.append(values)
            reference.append(basis_matrix(evaluation_x) @ view.reference_ivs)
    filtered_rms = float(np.sqrt(np.mean(np.diff(filtered, axis=0) ** 2)))
    reference_rms = float(np.sqrt(np.mean(np.diff(reference, axis=0) ** 2)))
    reduction = 1 - filtered_rms / reference_rms
    print(
        f"seed=430; default config; stationary jitter reduction={reduction:.1%}; "
        f"filtered RMS={filtered_rms:.6f}; reference RMS={reference_rms:.6f}; "
        f"90% level-step response={response_seconds}s"
    )
    assert reduction >= 0.30
    assert response_seconds is not None and response_seconds <= 10


def test_skew_shift_replay_is_finite_without_claiming_a_level_reset():
    manager = FilterManager()
    warm(manager)
    for i, x in enumerate([-1.4, -1.0, -0.6, 0, 0.6, 1.0, 1.4]):
        ingest(manager, x, iv=0.3 + 0.06 * x, now=1 + i * 0.01)
    view = manager.snapshots(now=1.1)[EXPIRY_MS]
    assert np.isfinite(view.knot_ivs).all()
    assert np.linalg.eigvalsh(view.covariance).min() >= -1e-12
    # A two-sided skew move is not promised to pass the same-sign level policy.
    assert any(d.status == "gated" for d in view.diagnostics) or view.resets > 0


def test_reference_only_forward_and_time_transport_preserves_the_total_variance_reference():
    manager = FilterManager(FilterConfig(process_iv_rate=0))
    originals = [quote(x, iv=0.3 + 0.02 * x) for x in [-1, 0]]
    for row, instrument in originals:
        manager.ingest(row, instrument, now_ms=NOW_MS, now=0)
    before = manager.snapshots(now=0)[EXPIRY_MS]
    shift = 0.1
    for row, instrument in originals:
        updated = SimpleNamespace(**vars(row))
        updated.forward *= math.exp(shift * CONTEXT.scale)
        updated.expiry *= 1.02
        manager.ingest(updated, instrument, now_ms=NOW_MS, now=1)
    view = manager.snapshots(now=1)[EXPIRY_MS]
    assert view.context.forward == pytest.approx(
        CONTEXT.forward * math.exp(shift * CONTEXT.scale)
    )
    assert view.context.exercise_time == pytest.approx(CONTEXT.exercise_time * 1.02)
    strikes = [quote(x)[0].strike for x in [-1, 0, 1]]
    expected_vols = [0.28, 0.3, math.sqrt(2 * 0.30**2 - 0.28**2)]
    assert view.volatility(
        view.context.coordinates(strikes), display=False
    ) == pytest.approx(expected_vols, abs=1e-5)
    assert view.updates == before.updates and view.resets == before.resets
    assert view.last_accepted_age == 1


def test_coordinate_overhang_reinitializes_with_explicit_reason():
    manager = FilterManager()
    warm(manager)
    for x in [-1.4, -1.0, -0.6, 0, 0.6, 1.0, 1.4]:
        row, instrument = quote(x)
        row.forward *= math.exp(0.6 * CONTEXT.scale)
        manager.ingest(row, instrument, now_ms=NOW_MS, now=1)
    view = manager.snapshots(now=1)[EXPIRY_MS]
    assert view.resets == 1 and view.reset_reason == "coordinate overhang"
    assert len(view.knot_ivs) == 9
    assert view.knot_ivs == pytest.approx([0.3] * 9)


def test_control_change_accounts_for_elapsed_interval_using_old_process_noise():
    manager = FilterManager()
    before = warm(manager)
    old_config = manager.config
    manager.configure(FilterConfig(process_iv_rate=0), now=2)
    expected = np.asarray(before.covariance) + process_covariance(old_config, 2)
    assert manager.snapshots(now=2)[EXPIRY_MS].covariance == pytest.approx(expected)
    manager.tick(now_ms=NOW_MS + 3000, now=3)
    assert manager.snapshots(now=3)[EXPIRY_MS].covariance == pytest.approx(expected)


def test_pending_clock_skewed_version_is_consumed_once_when_usable():
    manager = FilterManager()
    before = warm(manager)
    row, instrument = quote(0.6, iv=0.301, timestamp=NOW_MS + 100)
    manager.ingest(row, instrument, now_ms=NOW_MS, now=0)
    assert manager.snapshots(now=0)[EXPIRY_MS].updates == before.updates
    manager.tick(now_ms=NOW_MS + 100, now=0.1)
    assert manager.snapshots(now=0.1)[EXPIRY_MS].updates == before.updates
    manager.ingest(row, instrument, now_ms=NOW_MS + 150, now=0.15)
    assert manager.snapshots(now=0.15)[EXPIRY_MS].updates == before.updates + 1
    manager.ingest(row, instrument, now_ms=NOW_MS + 200, now=0.2)
    assert manager.snapshots(now=0.2)[EXPIRY_MS].updates == before.updates + 1


def test_timer_tick_does_not_bootstrap_without_events():
    manager = FilterManager()
    row, instrument = quote(0.0, iv=0.30, timestamp=NOW_MS + 500)
    manager.ingest(row, instrument, now_ms=NOW_MS, now=0)
    state = manager._filters[EXPIRY_MS]
    assert state.mean is None

    # Timer tick when quote is within time window must NOT bootstrap
    manager.tick(now_ms=NOW_MS + 600, now=0.6)
    assert state.mean is None
    assert row.instrument_name not in state.seen


def test_robust_and_adaptation_config_validation_matches_rust():
    for bad in (
        {"robust_sigma": 0.0},
        {"robust_sigma": math.nan},
        {"gate_sigma": 2.0},
        {"adapt_rate": 1.5},
        {"adapt_rate": -0.1},
        {"adapt_min_scale": 0.0},
        {"adapt_min_scale": 2.0},
        {"adapt_max_scale": 0.5},
    ):
        with pytest.raises(ValueError):
            FilterConfig(**bad)
    config = FilterConfig()
    assert config.robust_weight(2.0) == 1.0
    assert config.robust_weight(-4.0) == pytest.approx(0.75)
    assert config.adapt_rate == 0.0 and not config.regime_override


def _probe_z(delta):
    manager = FilterManager()
    warm(manager)
    ingest(manager, 0.6, iv=0.3 + delta, now=1)
    return next(
        d
        for d in manager.snapshots(now=1)[EXPIRY_MS].diagnostics
        if d.instrument == "q-0.6-c"
    )


def test_huber_weight_applies_between_robust_and_gate_sigma():
    z1 = _probe_z(0.01).innovation_z
    shifts = {}
    for target in (2.0, 4.0):
        diag = _probe_z(0.01 * target / z1)
        assert diag.status == "accepted"
        assert diag.innovation_z == pytest.approx(target, rel=1e-6)
        assert diag.weight == pytest.approx(min(1.0, 3.0 / target))
        shifts[target] = diag.innovation
    # A z=4 quote is down-weighted, so it moves the state by less than the
    # un-weighted gain would: compare with a config that disables Huber.
    plain = FilterManager(FilterConfig(robust_sigma=5.0))
    warm(plain)
    ingest(plain, 0.6, iv=0.3 + shifts[4.0], now=1)
    robust = FilterManager()
    warm(robust)
    ingest(robust, 0.6, iv=0.3 + shifts[4.0], now=1)
    plain_ivs = plain.snapshots(now=1)[EXPIRY_MS].knot_ivs
    robust_ivs = robust.snapshots(now=1)[EXPIRY_MS].knot_ivs
    assert abs(robust_ivs[5] - 0.3) < abs(plain_ivs[5] - 0.3)


def test_gate_sigma_is_configurable():
    z1 = _probe_z(0.01).innovation_z
    delta = 0.01 * 4.0 / z1
    manager = FilterManager(FilterConfig(gate_sigma=3.5))
    warm(manager)
    ingest(manager, 0.6, iv=0.3 + delta, now=1)
    statuses = [d.status for d in manager.snapshots(now=1)[EXPIRY_MS].diagnostics]
    assert "gated" in statuses


def test_regime_override_accepts_coherent_rejections_instead_of_resetting():
    manager = FilterManager(FilterConfig(regime_override=True))
    before = warm(manager)
    for i, x in enumerate([-1.0, 0.6, 1.0]):
        ingest(manager, x, iv=0.8, now=1 + i * 0.1)
    view = manager.snapshots(now=1.2)[EXPIRY_MS]
    assert view.resets == before.resets
    assert view.reset_reason == before.reset_reason
    accepted = [d for d in view.diagnostics if d.weight == 1.0 and d.innovation_z]
    assert len(accepted) >= 3 and all(d.status == "accepted" for d in accepted)
    assert min(view.knot_ivs[5], view.knot_ivs[6]) > 0.5
    state = manager._filters[EXPIRY_MS]
    assert state.process_scale == 1.0 and not state.outliers


def test_nis_adaptation_raises_process_scale_only_when_enabled():
    scales = {}
    for rate in (0.0, 0.1):
        rng = np.random.default_rng(7)
        manager = FilterManager(FilterConfig(adapt_rate=rate))
        for second in range(60):
            for i, x in enumerate([-1.4, -1, -0.6, 0, 0.6, 1, 1.4]):
                ingest(
                    manager,
                    x,
                    iv=0.3 + rng.normal(0, 0.012),
                    now=second + i * 0.001,
                )
        scales[rate] = manager._filters[EXPIRY_MS].process_scale
    assert scales[0.0] == 1.0
    assert scales[0.1] > 1.5
    assert scales[0.1] <= FilterConfig().adapt_max_scale


def test_nonfinite_innovation_is_marked_invalid_without_touching_the_state():
    manager = FilterManager()
    warm(manager)
    state = manager._filters[EXPIRY_MS]
    before = state.mean.copy()
    obs = state.observations[3]
    state.seen[obs.instrument] = obs.version - 1
    object.__setattr__(obs, "mid_iv", 1e308)
    manager._measure(state, [obs], set(), 1.0)
    assert state.diagnostics[obs.instrument].status == "invalid"
    assert np.array_equal(state.mean, before)


def test_regime_detector_feeds_process_noise_time_in_the_manager():
    config = FilterConfig(regime=RegimeConfig(threshold=4.0))
    manager = FilterManager(config)
    for second in range(120):
        for i, x in enumerate([-1.4, -1, -0.6, 0, 0.6, 1, 1.4]):
            ingest(manager, x, iv=0.3, now=second + i * 0.001)
    state = manager._filters[EXPIRY_MS]
    assert state.regime is not None and state.regime.multiplier == 1.0
    for second in range(120, 150):
        shocked = 0.3 + 0.08 * (second - 119) / 30
        for i, x in enumerate([-1.4, -1, -0.6, 0, 0.6, 1, 1.4]):
            ingest(manager, x, iv=shocked, now=second + i * 0.001)
    assert state.regime.multiplier > 5
    # Same elapsed time, bigger covariance growth than the default filter.
    assert manager._noise_dt(state, 1.0) > 5


def _wiggle(i):
    return math.modf(math.sin(i * 12.9898) * 43758.5453)[0] * 2 - 1


def _calm(det, start, n, vol, fwd, dt=10.0):
    t = start
    for i in range(n):
        det.observe(t, vol + 0.0002 * _wiggle(i), fwd * (1 + 1e-5 * _wiggle(i + 7)))
        t += dt
    return t


def test_regime_detector_calm_shock_and_exact_decay():
    det = RegimeDetector()
    t = _calm(det, 0.0, 1500, 0.40, 80_000.0)
    assert det.multiplier == 1.0
    peak = 1.0
    for k in range(1, 31):
        peak = max(peak, det.observe(t, 0.40 + 0.03 * k / 30, 80_000.0))
        t += 10.0
    assert 10 < peak <= det.config.max_multiplier
    at = {}
    for i in range(151):
        m = det.observe(t, 0.43, 80_000.0)
        if i in (40, 150):
            at[i] = m
        t += 10.0
    assert at[40] > 1.5
    ratio = (at[150] - 1) / (at[40] - 1)
    assert ratio == pytest.approx(math.exp(-1100 / det.config.decay), abs=1e-9)


def test_regime_detector_forward_shock_and_cap():
    det = RegimeDetector()
    t = _calm(det, 0.0, 600, 0.40, 80_000.0)
    assert det.observe(t, 0.40, 80_000.0 * 0.97) > 10
    det = RegimeDetector(RegimeConfig(max_multiplier=5.0))
    t = _calm(det, 0.0, 600, 0.40, 80_000.0)
    assert det.observe(t, 0.60, 80_000.0) == 5.0


def _after_jump(dt, calm_ticks, gap, jump):
    det = RegimeDetector()
    t = 0.0
    for i in range(calm_ticks):
        det.observe(t, 0.40 + 0.0001 * _wiggle(i), 80_000.0)
        t += dt
    return det.observe(t - dt + gap, 0.40 + jump, 80_000.0)


def test_regime_detector_sees_shocks_at_any_cadence_and_across_gaps():
    for dt in (10.0, 60.0, 120.0, 150.0, 180.0, 200.0, 250.0):
        assert _after_jump(dt, 120, dt, 0.10) > 10, dt
    assert _after_jump(10.0, 200, 301.0, 0.10) > 10
    assert _after_jump(10.0, 200, 3600.0, 0.10) > 10


def test_regime_detector_sparse_calm_and_flat_feeds_are_not_flagged():
    for dt in (60.0, 180.0, 301.0):
        det = RegimeDetector()
        t = 0.0
        for i in range(400):
            det.observe(t, 0.40 + 0.0001 * _wiggle(i), 80_000.0)
            t += dt
        assert det.multiplier == 1.0, dt
    det = RegimeDetector()
    t = 0.0
    for _ in range(72 * 360):
        det.observe(t, 0.40, 80_000.0)
        t += 10.0
    assert det.observe(t, 0.4003, 80_000.0) == 1.0


def test_regime_detector_validation_and_invalid_input_leave_state_unchanged():
    for bad in (
        {"window": 0.0},
        {"threshold": math.nan},
        {"decay": -1.0},
        {"max_multiplier": 0.5},
        {"initial_change_sd": 0.0},
    ):
        with pytest.raises(ValueError):
            RegimeConfig(**bad)
    det = RegimeDetector()
    _calm(det, 0.0, 50, 0.4, 80_000.0)
    snapshot = (list(det._history), det._change_var, det.multiplier, det._last_time)
    for args in (
        (math.nan, 0.4, 1e5),
        (1e4, -0.1, 1e5),
        (1e4, 0.4, 0.0),
        (1.0, 0.4, 1e5),
    ):
        with pytest.raises(ValueError):
            det.observe(*args)
    assert snapshot == (
        list(det._history),
        det._change_var,
        det.multiplier,
        det._last_time,
    )

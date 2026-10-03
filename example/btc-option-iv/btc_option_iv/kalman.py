"""In-memory, nine-knot Kalman smiles for visualization, not calibrated inference.

NumPy/SciPy are notebook extras; the streaming CLI does not import this module.
All IV quantities below use annualized decimal units and elapsed seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from itofin import ItofinError
from itofin.termstructures import TotalVarianceCubicSmileSection
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve

KNOTS = (-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0)
_KNOT_ARRAY = np.asarray(KNOTS)
_IDENTITY = np.eye(9)
_BASIS = CubicSpline(KNOTS, _IDENTITY, bc_type="natural", extrapolate=False)
MAX_QUOTE_AGE_MS = 10_000
MAX_OVERHANG = 0.25
MAX_END_SLOPE = 0.10


@dataclass(frozen=True)
class FilterConfig:
    """Experimental noise controls; measurement precision is inverse IV spread."""

    process_iv_rate: float = 0.001
    measurement_sd: float = 0.005
    spread_floor: float = 0.001
    reference_spread: float = 0.01
    correlation_length: float = 1.0
    independent_fraction: float = 0.1

    def __post_init__(self):
        for name in (
            "process_iv_rate",
            "measurement_sd",
            "spread_floor",
            "reference_spread",
            "correlation_length",
        ):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or value < 0
                or (name != "process_iv_rate" and value == 0)
            ):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if name == 'process_iv_rate' else 'positive'}"
                )
        if (
            not math.isfinite(self.independent_fraction)
            or not 0 < self.independent_fraction <= 1
        ):
            raise ValueError("independent_fraction must be in (0, 1]")

    def measurement_variance(self, spread: float) -> float:
        if not math.isfinite(spread) or spread < 0:
            raise ValueError("IV spread must be finite and nonnegative")
        variance = (
            self.measurement_sd**2
            * max(spread, self.spread_floor)
            / self.reference_spread
        )
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("measurement variance must be finite and positive")
        return variance


@dataclass(frozen=True)
class SmileContext:
    forward: float
    exercise_time: float
    atm_vol: float

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.forward, self.exercise_time, self.atm_vol)
        ):
            raise ValueError("smile context must be finite and positive")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("invalid ATM standard deviation")
        try:
            endpoints = [
                math.exp(math.log(self.forward) + z * self.scale) for z in (-3, 3)
            ]
        except OverflowError as exc:
            raise ValueError("non-finite knot strike") from exc
        if any(not math.isfinite(k) or k <= 0 for k in endpoints):
            raise ValueError("knot strikes must be finite and positive")

    @property
    def scale(self) -> float:
        return self.atm_vol * math.sqrt(self.exercise_time)

    def coordinates(self, strikes):
        strikes = np.asarray(strikes, dtype=float)
        if not np.isfinite(strikes).all() or (strikes <= 0).any():
            raise ValueError("strikes must be finite and positive")
        x = (np.log(strikes) - math.log(self.forward)) / self.scale
        # Strike-space tolerance agrees with the core's low-ATM round-trip fix.
        for endpoint in (KNOTS[0], KNOTS[-1]):
            boundary = math.exp(math.log(self.forward) + endpoint * self.scale)
            x = np.where(np.isclose(strikes, boundary, rtol=1e-13, atol=0), endpoint, x)
        return x


def basis_matrix(x):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.ndim != 1 or not np.isfinite(x).all() or (x < -3).any() or (x > 3).any():
        raise ValueError("basis queries must be finite and inside the knot domain")
    return np.asarray(_BASIS(x))


def process_covariance(config: FilterConfig, dt: float):
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    correlation = np.exp(
        -np.abs(_KNOT_ARRAY[:, None] - _KNOT_ARRAY) / config.correlation_length
    )
    result = (
        config.process_iv_rate**2
        * dt
        * (
            (1 - config.independent_fraction) * correlation
            + config.independent_fraction * _IDENTITY
        )
    )
    if not np.isfinite(result).all():
        raise ValueError("non-finite process covariance")
    return result


def _validate_state(mean, covariance):
    if mean.shape != (9,) or covariance.shape != (9, 9):
        raise ValueError("expected nine knot IVs and a 9-by-9 covariance")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("non-finite filter state")
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance):
        raise ValueError("asymmetric covariance")
    if np.linalg.eigvalsh(covariance).min() < -tolerance:
        raise ValueError("covariance is not positive semidefinite")


def kalman_update(mean, covariance, h, y, variances):
    """Cholesky gain solve and Joseph update; never invert the innovation matrix."""
    mean, covariance = (
        np.asarray(mean, dtype=float),
        np.asarray(covariance, dtype=float),
    )
    h, y, variances = (
        np.asarray(h, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(variances, dtype=float),
    )
    _validate_state(mean, covariance)
    if (
        y.ndim != 1
        or len(y) == 0
        or h.shape != (len(y), 9)
        or variances.shape != y.shape
    ):
        raise ValueError("invalid measurement dimensions")
    if (
        not all(np.isfinite(v).all() for v in (h, y, variances))
        or (variances <= 0).any()
    ):
        raise ValueError("invalid measurement values or variances")
    innovation = y - h @ mean
    s = h @ covariance @ h.T + np.diag(variances)
    gain = cho_solve(cho_factor(s, lower=True), h @ covariance).T
    updated = mean + gain @ innovation
    a = _IDENTITY - gain @ h
    p = a @ covariance @ a.T + (gain * variances) @ gain.T
    p = (p + p.T) / 2
    _validate_state(updated, p)
    return updated, p, innovation, innovation / np.sqrt(np.diag(s))


def transport(mean, covariance, old: SmileContext, new: SmileContext):
    """Transport the strike-space curve, with bounded linear tails and a Jacobian."""
    mean, covariance = (
        np.asarray(mean, dtype=float),
        np.asarray(covariance, dtype=float),
    )
    _validate_state(mean, covariance)
    if old == new:
        return mean.copy(), covariance.copy(), _IDENTITY.copy()
    u = (
        math.log(new.forward) - math.log(old.forward) + _KNOT_ARRAY * new.scale
    ) / old.scale
    distances = np.maximum(np.maximum(-3 - u, u - 3), 0)
    if not np.isfinite(u).all() or distances.max() > MAX_OVERHANG + 1e-12:
        raise ValueError("coordinate overhang exceeds 0.25 standardized units")
    j = basis_matrix(np.clip(u, -3, 3))
    values = j @ mean
    for i in np.flatnonzero(distances > 0):
        endpoint = -3.0 if u[i] < -3 else 3.0
        delta = u[i] - endpoint
        derivative = np.asarray(_BASIS(endpoint, 1))
        slope = float(derivative @ mean)
        values[i] += float(np.clip(slope, -MAX_END_SLOPE, MAX_END_SLOPE)) * delta
        # At the boundary, use the unclamped one-sided derivative.
        if abs(slope) <= MAX_END_SLOPE + 1e-14:
            j[i] += delta * derivative
    p = j @ covariance @ j.T + np.diag((0.01 * distances / MAX_OVERHANG) ** 2)
    p = (p + p.T) / 2
    _validate_state(values, p)
    return values, p, j


@dataclass(frozen=True)
class Observation:
    instrument: str
    option_type: str
    strike: float
    timestamp_ms: int
    mid_iv: float
    bid_iv: float
    ask_iv: float
    forward: float
    exercise_time: float
    version: int
    alternatives: int = 1

    @property
    def spread(self):
        return self.ask_iv - self.bid_iv


@dataclass(frozen=True)
class Diagnostic:
    instrument: str
    version: int
    status: str
    innovation: float | None = None
    innovation_z: float | None = None


@dataclass(frozen=True)
class SmileSnapshot:
    context: SmileContext | None
    knot_ivs: tuple[float, ...] | None
    covariance: tuple[tuple[float, ...], ...] | None
    observations: tuple[Observation, ...]
    diagnostics: tuple[Diagnostic, ...]
    reference_ivs: tuple[float, ...] | None
    reference_error: str | None
    status: str
    last_accepted_age: float | None
    reset_reason: str
    resets: int
    updates: int
    warnings: tuple[str, ...]
    observed_support: tuple[float, float] | None = None
    reference_section: TotalVarianceCubicSmileSection | None = field(
        default=None, compare=False
    )

    @property
    def segment_coefficients(self):
        if self.knot_ivs is None:
            return ()
        c = CubicSpline(KNOTS, self.knot_ivs, bc_type="natural").c
        return tuple(
            tuple(float(v) for v in row) for row in np.column_stack((c[2], c[1], c[0]))
        )

    def volatility(self, x, *, display=True):
        if self.knot_ivs is None:
            raise ValueError("filter is waiting for bootstrap observations")
        values = basis_matrix(x) @ np.asarray(self.knot_ivs)
        return np.maximum(values, 0) if display else values


@dataclass
class _ExpiryState:
    context: SmileContext | None = None
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    clock: float | None = None
    last_accepted: float | None = None
    seen: dict[str, int] = field(default_factory=dict)
    quarantined: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, Diagnostic] = field(default_factory=dict)
    outliers: dict[float, tuple[float, int]] = field(default_factory=dict)
    observations: tuple[Observation, ...] = ()
    support: tuple[float, float] | None = None
    reset_reason: str = "bootstrap"
    resets: int = 0
    updates: int = 0
    warning: str | None = None


def _context(observations):
    if not observations:
        raise ValueError("no fresh usable quotes")
    forward = float(np.median([o.forward for o in observations]))
    t = float(np.median([o.exercise_time for o in observations]))
    positive = [o for o in observations if o.mid_iv > 0]
    if not positive:
        raise ValueError("no positive ATM reference IV")
    atm = min(
        positive, key=lambda o: (abs(math.log(o.strike / forward)), o.instrument)
    ).mid_iv
    return SmileContext(forward, t, atm)


def _reference(context, observations):
    smile = TotalVarianceCubicSmileSection(
        [o.strike for o in observations],
        [o.mid_iv for o in observations],
        context.forward,
        context.exercise_time,
        context.atm_vol,
    )
    strikes = [math.exp(math.log(context.forward) + x * context.scale) for x in KNOTS]
    values = tuple(smile.volatility(k) for k in strikes)
    if len(values) != 9 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid regularized-fit ordinates")
    return values, smile


class FilterManager:
    """Own all expiry filters; calls belong to the feed/controller, never UI reads.

    Premium-content versions ignore timestamps, amounts, and reference-only IV
    revaluations. An initially unusable premium can be consumed once when its
    model inputs first become available. Returning prices A -> B -> A are new
    versions; changing call/put selection does not replay an old version.
    """

    def __init__(self, config=None):
        self.config = config or FilterConfig()
        self._filters: dict[int, _ExpiryState] = {}
        self._quotes: dict[int, dict[str, Observation]] = {}
        self._versions: dict[tuple[int, str], tuple[int, tuple, int]] = {}

    def ingest(self, row, instrument, *, now_ms: int, now: float):
        expiry = instrument.expiration_timestamp_ms
        if expiry <= now_ms:
            return
        state = self._filters.setdefault(expiry, _ExpiryState())
        if not math.isfinite(now) or (state.clock is not None and now < state.clock):
            raise ValueError("monotonic filter clock moved backward")
        cache = self._quotes.setdefault(expiry, {})
        name = row.instrument_name
        key = (expiry, name)
        previous = self._versions.get(key)
        if previous is not None and row.timestamp_ms < previous[0]:
            return
        try:
            values = (
                row.mid_iv,
                row.bid_iv,
                row.ask_iv,
                row.strike,
                row.forward,
                row.expiry,
            )
            if any(v is None or not math.isfinite(v) for v in values):
                raise ValueError("missing/non-finite IV inputs")
            if (
                min(row.mid_iv, row.bid_iv, row.ask_iv) < 0
                or min(row.strike, row.forward, row.expiry) <= 0
            ):
                raise ValueError("invalid IV/context inputs")
            if (
                row.ask_iv < row.bid_iv
                or not row.bid_iv - 1e-12 <= row.mid_iv <= row.ask_iv + 1e-12
            ):
                raise ValueError("crossed/inconsistent IV spread")
            if row.timestamp_ms > now_ms + 1000:
                raise ValueError("future-dated quote")
            # Real IvRow premiums are available. This also keeps model-reference
            # changes from masquerading as independent new option measurements.
            fingerprint = (row.bid_premium_btc, row.ask_premium_btc)
            if any(p is None or not math.isfinite(p) or p < 0 for p in fingerprint):
                raise ValueError("invalid source premiums")
            changed = previous is None or fingerprint != previous[1]
            version = (previous[2] if previous else 0) + int(changed)
            self._versions[key] = (row.timestamp_ms, fingerprint, version)
            cache[name] = Observation(
                name,
                instrument.option_type.lower(),
                row.strike,
                row.timestamp_ms,
                row.mid_iv,
                row.bid_iv,
                row.ask_iv,
                row.forward,
                row.expiry,
                version,
            )
            # Only unseen premium versions may be consumed. This permits a
            # pending version to become usable once (e.g. after clock skew or
            # missing references), but never reuses already-consumed premiums.
            self._refresh(expiry, now_ms=now_ms, now=now, measurements=True)
        except (ValueError, OverflowError, np.linalg.LinAlgError) as exc:
            cache.pop(name, None)
            state.warning = f"Rejected quote {name}: {exc}"
            self._refresh(expiry, now_ms=now_ms, now=now, measurements=False)

    def discard(self, name, expiry):
        self._quotes.get(expiry, {}).pop(name, None)

    def _select(self, expiry, now_ms):
        groups = {}
        for o in self._quotes.get(expiry, {}).values():
            if 0 <= now_ms - o.timestamp_ms <= MAX_QUOTE_AGE_MS:
                groups.setdefault(o.strike, []).append(o)
        selected = []
        for strike, candidates in sorted(groups.items()):

            def rank(o):
                otm = (o.option_type == "c" and o.strike >= o.forward) or (
                    o.option_type == "p" and o.strike <= o.forward
                )
                return o.spread, not otm, o.instrument

            selected.append(
                replace(min(candidates, key=rank), alternatives=len(candidates))
            )
        return tuple(selected)

    @staticmethod
    def _reset(state, reason):
        state.mean = state.covariance = state.context = None
        state.last_accepted = None
        state.seen.clear()
        state.diagnostics.clear()
        state.outliers.clear()
        if reason != "coordinate overhang":
            state.quarantined.clear()
        state.reset_reason = reason
        state.resets += 1
        state.warning = None

    def _bootstrap(self, state, now):
        usable = [
            o
            for o in state.observations
            if state.quarantined.get(o.instrument) != o.version
        ]
        try:
            context = _context(usable)
            values, _ = _reference(context, usable)
            values = np.asarray(values)
            covariance = _IDENTITY * 0.05**2
            _validate_state(values, covariance)
            used = [o for o in usable if -3 <= context.coordinates([o.strike])[0] <= 3]
            state.context, state.mean, state.covariance = context, values, covariance
            state.clock = state.last_accepted = now
            state.warning = None
            for o in used:
                state.seen[o.instrument] = o.version
                state.diagnostics[o.instrument] = Diagnostic(
                    o.instrument, o.version, "bootstrap"
                )
        except (ValueError, OverflowError, ItofinError, np.linalg.LinAlgError) as exc:
            state.warning = f"Waiting for bootstrap: {exc}"

    def _refresh(self, expiry, *, now_ms, now, measurements):
        state = self._filters[expiry]
        if not math.isfinite(now) or (state.clock is not None and now < state.clock):
            raise ValueError("monotonic filter clock moved backward")
        state.observations = self._select(expiry, now_ms)
        if state.observations:
            state.support = (
                state.observations[0].strike,
                state.observations[-1].strike,
            )
        if state.last_accepted is not None and now - state.last_accepted >= 60:
            self._reset(state, "long feed gap")
        if state.mean is None:
            if measurements:
                self._bootstrap(state, now)
            return
        dt = now - state.clock
        changed = (
            [
                o
                for o in state.observations
                if state.seen.get(o.instrument, 0) < o.version
            ]
            if measurements
            else []
        )
        pre_gated = set()
        try:
            # Gate fresh source IVs against the old strike-space prior BEFORE
            # using their ATM IVs as normalization. An isolated ATM spike must
            # not evade the innovation gate by inducing a coordinate reset.
            preview_p = state.covariance + process_covariance(self.config, dt)
            for o in changed:
                x = state.context.coordinates([o.strike])[0]
                if -3 <= x <= 3:
                    h = basis_matrix([x])[0]
                    z = (o.mid_iv - h @ state.mean) / math.sqrt(
                        float(
                            h @ preview_p @ h
                            + self.config.measurement_variance(o.spread)
                        )
                    )
                    if abs(z) > 5:
                        pre_gated.add(o.instrument)
            trusted = [
                o
                for o in state.observations
                if o.instrument not in pre_gated
                and state.quarantined.get(o.instrument) != o.version
            ]
            context = _context(trusted) if trusted else state.context
            try:
                values, covariance, _ = transport(
                    state.mean, state.covariance, state.context, context
                )
            except ValueError as exc:
                if "overhang" not in str(exc):
                    raise
                for o in changed:
                    if o.instrument in pre_gated:
                        state.quarantined[o.instrument] = o.version
                self._reset(state, "coordinate overhang")
                self._bootstrap(state, now)
                return
            covariance += process_covariance(self.config, dt)
            _validate_state(values, covariance)
            state.context, state.mean, state.covariance, state.clock = (
                context,
                values,
                covariance,
                now,
            )
            if changed:
                self._measure(state, changed, pre_gated, now)
        except (ValueError, OverflowError, np.linalg.LinAlgError) as exc:
            state.warning = f"Numerical update skipped: {exc}"

    def _measure(self, state, changed, pre_gated, now):
        eligible = []
        xs = []
        for o in changed:
            x = state.context.coordinates([o.strike])[0]
            if -3 <= x <= 3:
                eligible.append(o)
                xs.append(x)
            else:
                state.diagnostics[o.instrument] = Diagnostic(
                    o.instrument, o.version, "outside"
                )
        if not eligible:
            return
        h = basis_matrix(xs)
        y = np.asarray([o.mid_iv for o in eligible])
        r = np.asarray([self.config.measurement_variance(o.spread) for o in eligible])
        innovations = y - h @ state.mean
        variances = np.einsum("ij,jk,ik->i", h, state.covariance, h) + r
        z = innovations / np.sqrt(variances)
        if not np.isfinite(z).all():
            raise ValueError("invalid innovation statistics")
        gated = (np.abs(z) > 5) | np.asarray(
            [o.instrument in pre_gated for o in eligible]
        )
        state.outliers = {
            strike: entry
            for strike, entry in state.outliers.items()
            if now - entry[0] <= 3
        }
        for i, o in enumerate(eligible):
            state.seen[o.instrument] = o.version
            state.diagnostics[o.instrument] = Diagnostic(
                o.instrument,
                o.version,
                "gated" if gated[i] else "accepted",
                float(innovations[i]),
                float(z[i]),
            )
            if gated[i]:
                state.quarantined[o.instrument] = o.version
                state.outliers[o.strike] = (now, 1 if innovations[i] > 0 else -1)
            else:
                state.quarantined.pop(o.instrument, None)
                state.outliers.pop(o.strike, None)
        if any(
            sum(entry[1] == sign for entry in state.outliers.values()) >= 3
            for sign in (-1, 1)
        ):
            self._reset(state, "multi-strike regime change")
            self._bootstrap(state, now)
            return
        accepted = ~gated
        if accepted.any():
            try:
                values, covariance, _, _ = kalman_update(
                    state.mean, state.covariance, h[accepted], y[accepted], r[accepted]
                )
            except (ValueError, np.linalg.LinAlgError):
                for o, accept in zip(eligible, accepted):
                    if accept:
                        state.diagnostics[o.instrument] = Diagnostic(
                            o.instrument, o.version, "solver_failed"
                        )
                raise
            state.mean, state.covariance = values, covariance
            state.last_accepted = now
            state.updates += int(accepted.sum())
            state.warning = None

    def tick(self, *, now_ms, now):
        self.retire(set(self._filters), now_ms=now_ms)
        for expiry in list(self._filters):
            self._refresh(expiry, now_ms=now_ms, now=now, measurements=False)

    def configure(self, config: FilterConfig, *, now):
        # Account for the interval before a control change using the OLD noise.
        for state in self._filters.values():
            if state.mean is not None:
                if now < state.clock:
                    raise ValueError("monotonic filter clock moved backward")
                covariance = state.covariance + process_covariance(
                    self.config, now - state.clock
                )
                _validate_state(state.mean, covariance)
                state.covariance, state.clock = covariance, now
        self.config = config

    def reset_all(self, *, now_ms, now):
        for expiry, state in self._filters.items():
            self._reset(state, "manual reset")
            self._refresh(expiry, now_ms=now_ms, now=now, measurements=True)

    def retire(self, active_expiries, *, now_ms):
        keep = {e for e in active_expiries if e > now_ms}
        self._filters = {e: v for e, v in self._filters.items() if e in keep}
        self._quotes = {e: v for e, v in self._quotes.items() if e in keep}
        self._versions = {key: v for key, v in self._versions.items() if key[0] in keep}

    def snapshots(self, *, now):
        """Detached immutable views. Reading them NEVER assimilates or predicts."""
        result = {}
        for expiry, state in self._filters.items():
            age = (
                max(0.0, now - state.last_accepted)
                if state.last_accepted is not None
                else None
            )
            status = (
                "waiting"
                if state.mean is None
                else ("stale" if age >= 10 else "tracking")
            )
            warnings = [state.warning] if state.warning else []
            reference, reference_section, reference_error = None, None, None
            if state.mean is not None:
                # Include raw/gated quotes in the reference comparison, not in
                # the trusted metadata calculation for the filtered context.
                try:
                    reference, reference_section = _reference(
                        state.context, state.observations
                    )
                except (ValueError, OverflowError, ItofinError) as exc:
                    reference_error = str(exc)
                spline = CubicSpline(KNOTS, state.mean, bc_type="natural")
                roots = spline.derivative().roots(extrapolate=False)
                extrema = np.concatenate((_KNOT_ARRAY, roots[np.isfinite(roots)]))
                if (spline(extrema) < 0).any():
                    warnings.append(
                        "Negative IV/overshoot: display-only flooring; internal state unchanged."
                    )
            diagnostics = []
            for o in state.observations:
                diagnostic = state.diagnostics.get(o.instrument)
                if diagnostic is None or diagnostic.version != o.version:
                    diagnostic = Diagnostic(o.instrument, o.version, "unchanged")
                diagnostics.append(diagnostic)
            result[expiry] = SmileSnapshot(
                state.context,
                tuple(float(v) for v in state.mean) if state.mean is not None else None,
                tuple(tuple(float(v) for v in row) for row in state.covariance)
                if state.mean is not None
                else None,
                state.observations,
                tuple(diagnostics),
                reference,
                reference_error,
                status,
                age,
                state.reset_reason,
                state.resets,
                state.updates,
                tuple(warnings),
                state.support,
                reference_section,
            )
        return result

# Cubic mid-IV smile on standardized moneyness

## Goal

Provide a reusable `CubicSmileSection` for one expiry, fitted exactly through
market mid implied volatilities and queried on a dimensionless standard-
deviation axis. Expose the same implementation through Rust and Python so a
market-data client can turn one expiration's strike/`mid_iv` rows into a
repeatable smile curve.

## Coordinate and fit

The input is one forward `F`, one positive exercise time `T` in years, one
positive ATM reference volatility `sigma_atm`, and paired option strikes `K_i`
and mid implied volatilities `sigma_i`. Volatilities are annualized decimals.
For every observation, calculate

```text
x_i = (ln(K_i) - ln(F)) / (sigma_atm * sqrt(T))
```

`x_i` is signed standard-deviation log-moneyness: zero is ATM, positive is the
upper-strike/call wing, and negative is the lower-strike/put wing. `sigma_atm`
is fixed for the section; it is not recomputed from each observation or
iterated during fitting.

Sort the `(x_i, sigma_i)` pairs by `x_i` and construct the existing natural
cubic spline (`CubicDerivativeApprox::Spline`, zero second derivative at both
ends). The spline passes exactly through every input mid-IV node. This is
interpolation, not a weighted regression or smoothing fit. Call and put quotes
at the same strike must be consolidated by the caller before construction;
duplicate strikes are rejected rather than silently averaged.

The default **sample/evaluation** points are:

```text
[-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0]
```

These are output sample positions, not spline knots. Callers may replace the
sample grid. `sampled_mid_ivs` returns `None` for a sample outside the observed
node range unless extrapolation was explicitly enabled. Direct interpolation
queries outside the node range return an error by default. Explicit
extrapolation extends the end cubic segments and is for display/use only; it is
not an arbitrage guarantee.

## API

Core Rust type: `libitofin::termstructures::volatility::CubicSmileSection`.
Python type: `itofin.termstructures.CubicSmileSection`.

Construction takes `(strikes, mid_ivs, forward, exercise_time, atm_vol)`;
Python additionally accepts optional `std_dev_points=None` and
`extrapolate=False`. The core provides default-grid construction plus builders
for a custom sample grid and extrapolation. Queries include volatility at a
strike, volatility at a standard-deviation point, strike at a standard-
deviation point, the sorted source nodes, and sampled values on the configured
grid. It implements `SmileSection`, with the forward as its ATM level, so the
existing variance and Black option-price helpers remain available.

## Validation and failure behavior

- Require equal strike/IV lengths and at least two observations.
- Require finite positive `forward`, `exercise_time`, and `atm_vol`.
- Require finite positive strikes and finite nonnegative mid-IVs.
- Require finite sample points; preserve their caller-specified order.
- Sort source observations by standardized coordinate; reject duplicate
  coordinates (including duplicate strikes) through the spline's strict-node
  validation.
- Reject invalid or out-of-domain direct queries unless explicit
  extrapolation is enabled. Floor negative spline overshoot to zero when
  returning a volatility, consistent with the existing cubic optionlet smile.
- Do not silently drop missing/non-finite market observations: callers must
  filter unusable quote rows before construction.

## Non-goals

No bid/ask fitting, quote weighting, duplicate-strike aggregation, expiry
interpolation, live quote observation, implied-volatility inversion,
volatility recalibration, smoothing penalty, or static-arbitrage constraints.
The caller owns quote quality, the ATM-vol reference, and conversion of calls
and puts to a unique strike grid. A natural cubic spline can violate butterfly
arbitrage and must not be represented as an arbitrage-free surface.

## Validation

- Core tests cover coordinate/strike round trips, unsorted data, exact node
  recovery, linear-smile recovery, defaults and custom sample grids, out-of-
  range behavior, explicit extrapolation, duplicate nodes, and invalid inputs.
- Python binding tests cover construction, properties, query methods, optional
  arguments, and surfaced `ItofinError`s.
- Regenerate PyO3 stubs and run focused Rust and Python tests.

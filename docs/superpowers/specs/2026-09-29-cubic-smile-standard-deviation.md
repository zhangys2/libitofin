# Cubic mid-IV smile on standardized moneyness

## Goal

Provide a reusable `CubicSmileSection` for one expiry, fitted by curvature-
regularized least squares to market mid implied volatilities on a fixed standard-deviation
knot grid. Expose the same implementation through Rust and Python so a market-
data client can turn one expiration's strike/`mid_iv` rows into a repeatable
smile curve.

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

Sort the `(x_i, sigma_i)` observations by `x_i`. The configured knot
coordinates define one natural cubic spline (`CubicDerivativeApprox::Spline`,
zero second derivative at both ends); fit its knot IV ordinates to the source
observations by curvature-regularized least squares. The spline has no knots beyond the
configured set. Observations outside the knot range are excluded from fitting
rather than extrapolated. Call and put quotes at the same strike must be
consolidated by the caller before construction; duplicate source coordinates
are rejected rather than silently averaged.

The objective before nonnegative knot clipping is:

```text
mean_i((s(x_i) - sigma_i)^2) + smoothing * integral(s''(x)^2 dx)
```

The integral spans the fixed knot domain. `smoothing` defaults to **0.01**, is
finite and nonnegative, and acts in standardized x units. This is an explicit
visualization default, not a statistically calibrated parameter. Increasing it
favors smoother fits. Using a mean data term keeps the penalty weight comparable
as quote counts change. The implementation augments the data matrix with
zero-target, two-point Gauss-Legendre curvature rows on each segment, then uses
pivoted QR. Since each segment's second derivative is linear, this quadrature
integrates squared curvature exactly, including uneven knot spacing.

Positive smoothing has an affine nullspace: two distinct in-range observations
fix level and slope, so sparse data can determine all nine knot IVs. Unsampled
regions remain model-dependent; with only two observations the fit is linear.
No fake observations or implied ATM anchors are inserted. `smoothing=0` restores
unregularized fitting and requires at least as many observations as knots plus
full observation rank. Flooring negative fitted knot ordinates is a post-fit
visualization safeguard, not a nonnegative constrained optimizer.

The default **knot** points are:

```text
[-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0]
```

These are the exact nine spline knots; there are no additional knots at the
observed strikes. Callers may replace the knot grid, which is sorted before
fitting. `sampled_mid_ivs` returns fitted ordinates at those knots. Direct
queries outside the knot range return an error by default. Explicit
extrapolation extends the end cubic segments and is for display/use only; it is
not an arbitrage guarantee.

## API

Core Rust type: `libitofin::termstructures::volatility::CubicSmileSection`.
Python type: `itofin.termstructures.CubicSmileSection`.

Construction takes `(strikes, mid_ivs, forward, exercise_time, atm_vol)`;
Python additionally accepts optional `std_dev_points=None`,
`extrapolate=False`, and `smoothing=0.01`. The core provides default-grid
construction, custom knots, `with_smoothing`, and extrapolation. Go exposes
`Session.CubicSmileSectionWithSmoothing` and a `Smoothing` getter; the original
C/Go constructors retain their signatures and use the default penalty.
At least two distinct in-range observations are required with positive
smoothing, and the augmented design must have full rank. Fitted knot IVs are
floored at zero. Queries include volatility at a strike,
volatility at a standard-deviation point, strike at a standard-deviation point,
the sorted source observations, fitted knot values, and values on the knot
grid. Read-only parameters include the forward, expiry time, ATM volatility, smoothing,
and each segment's local polynomial coefficients `[a, b, c]` for
`sigma(x) = sigma_i + a*dx + b*dx^2 + c*dx^3`, where `dx = x - x_i`.
Knot residuals are zero up to floating-point rounding. Observation residuals
are fitted minus observed IV; out-of-range observations have no residual. They
are not estimates of quote uncertainty. The type implements `SmileSection`,
with the forward as its ATM level, so the existing variance and Black
option-price helpers remain available.

## Validation and failure behavior

- Require equal strike/IV lengths and at least two observations.
- Require finite positive `forward`, `exercise_time`, and `atm_vol`.
- Require finite positive strikes and finite nonnegative mid-IVs.
- Require at least two distinct finite knot points; sort them into increasing
  order.
- Sort source observations by standardized coordinate; reject duplicate
  coordinates (including duplicate strikes).
- Require finite nonnegative smoothing. With positive smoothing require at
  least two distinct in-range observations; with zero require at least as many
  as knots. Require a full-rank augmented fit in either case.
  Observations beyond the fixed knot range are retained for reporting but do not
  affect the fit.
- Reject invalid or out-of-domain direct queries unless explicit
  extrapolation is enabled. Floor negative fitted knot ordinates and cubic
  overshoot at zero when returning a volatility.
- Do not silently drop missing/non-finite market observations: callers must
  filter unusable quote rows before construction.

## Non-goals

No bid/ask fitting, quote weighting, duplicate-strike aggregation, expiry
interpolation, live quote observation, implied-volatility inversion,
volatility recalibration, automatic smoothing selection, or static-arbitrage constraints.
The caller owns quote quality, the ATM-vol reference, and conversion of calls
and puts to a unique strike grid. A natural cubic spline can violate butterfly
arbitrage and must not be represented as an arbitrage-free surface.

## Validation

- Core tests cover coordinate/strike round trips, unsorted observations,
  default/custom fixed knots, linear-smile recovery, coefficient reconstruction,
  zero knot residuals versus nonzero observation residuals, out-of-range behavior,
  explicit extrapolation, duplicate inputs, sparse two/seven-quote fits,
  closed-form curvature-penalty recovery, unequal-spacing smoothing, and
  invalid/rank-deficient fits.
- Python binding tests cover construction, parameters, residuals, query methods,
  optional arguments, and surfaced `ItofinError`s.
- Regenerate PyO3 stubs and run focused Rust and Python tests.

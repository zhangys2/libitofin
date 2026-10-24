# Kalman updates for the fixed-knot cubic smile

**Status:** User-approved design; Python visualization prototype implemented on
2026-09-30. Numerical defaults remain experimental, not market calibrations.
The implementation clarifications and validation results are recorded below.

## Goal and scope

Reduce visual jitter in the live BTC option smile while following genuine market
moves. This is a visualization prototype, not a trading signal, arbitrage-free
model, or calibrated uncertainty estimator.

- Implement in Python using NumPy/SciPy in `example/btc-option-iv/` first.
- Maintain independent filter state for every active expiry, not just the expiry
  selected in the reader. Switching the dropdown must not reset a filter.
- Keep state in memory. No restoration across app/server restarts.
- Filter the nine knot IVs; derive natural-cubic segment coefficients afterward.
- Assimilate fresh market mid-IV observations, not the fitted knot IVs of the
  existing regularized smile. Fitted knot estimates are correlated and must not
  be mistaken for nine independent market measurements.
- Overlay the current regularized quote fit and the filtered smile. Keep plots
  within observed support and the fixed knot domain.
- Do not change the core Rust cubic fitter or require a reusable Rust Kalman API
  in this prototype.

The existing static fit is documented in
[the cubic-smile specification](2026-09-29-cubic-smile-standard-deviation.md).

## State and observation model

For each expiry, retain a context `(F, T, sigma_atm)` and the fixed locations

```text
z = [-3, -1.5, -1, -0.6, 0, 0.6, 1, 1.5, 3]
scale = sigma_atm * sqrt(T)
K_j = exp(log(F) + z_j * scale)
```

`v` is a nine-element vector of annualized decimal IVs at these knots. `P` is
its 9-by-9 covariance. The natural cubic reconstructed from `v` has zero second
derivative at both ends. There are still eight segments, regardless of how many
market strikes are currently available.

Let `b_j(x)` be the natural cubic basis obtained by assigning ordinate 1 at knot
`j` and 0 at every other knot. Interpolation is linear in the ordinates:

```text
s(x) = sum_j b_j(x) * v_j
x_i = (log(K_i) - log(F)) / scale
H[i, j] = b_j(x_i)
y_i = observed market mid IV
innovation = y - H * v_predicted
```

Use the same natural-cubic basis for measurement evaluation, state transport,
curve evaluation, and coefficient reconstruction. Do not filter each segment's
coefficients independently: doing so could violate spline continuity.

Only observations within the current knot range enter measurement updates.
Outside quotes remain available for inspection, but are not extrapolated into
measurements. A running filter can update with one eligible quote; bootstrap
still requires at least two distinct in-range strikes.

### Normalization context

Preserve the example's current positive forward, exercise-time, and ATM-reference
selection conventions initially: median forward/time from the usable expiry
rows and positive mid-IV nearest the forward for the ATM reference. Apply these
conventions consistently to the selected input set. Do not obtain normalization
from the filtered ATM ordinate, which would introduce a circular observation
model.

Changes to forward, exercise time, and ATM reference are coordinate changes:
transport the prior before interpreting new observations. They are not, by
themselves, nine new IV measurements.

Integration clarification: pre-gate unseen premium versions against the old
strike-space prior before their ATM IV can change normalization; otherwise an
isolated ATM spike could evade gating through a coordinate reset. Exclude gated
versions from trusted context selection until their premium version changes or
a reset explicitly clears quarantine. The current reference fit still uses the
raw selected quotes (including gated ones) in the **same trusted context**, so
both curves remain comparable.

## Measurement selection and precision

Within an expiry, select one fresh call/put observation per strike: prefer the
quote with the tighter solved IV spread. Do not count the call and put as two
independent measurements of the same strike. The deterministic tie-break is OTM preference followed by instrument name;
both are documented and tested.

Use decimal IV width, not BTC premium width:

```text
w_i = ask_iv_i - bid_iv_i
w_effective_i = max(w_i, spread_floor)
R[i, i] = sigma_ref^2 * w_effective_i / reference_spread
R[i, j] = 0 for i != j
```

Therefore measurement precision `1/R[i,i]` is proportional to **inverse spread**,
not inverse squared spread. `sigma_ref` is the assumed measurement standard
deviation at `reference_spread`; it is a tuning parameter, not a confidence
estimate extracted from the quote.

Reject non-finite values and negative/crossed spreads. Zero spread is admissible
only with the positive floor. The measurement-noise control must remain strictly
positive. Diagonal `R` is an explicit prototype approximation: shared model
inputs and market behavior can correlate errors across strikes.

### Freshness and event identity

Assimilate at quote-event cadence; refresh plots at the existing three-second
cadence. Filter updates must not be driven by marimo rendering/reactivity.

- Maintain per-expiry measurement identity/version tracking outside UI cells.
- A redisplayed snapshot, duplicate source event, or unchanged selected quote
  must not be assimilated repeatedly. Advancing a heartbeat timestamp alone is
  not independent new evidence.
- Refresh quote freshness bookkeeping even when measurement content is unchanged.
- Switching from a selected quote to another must not replay a previously
  consumed, unchanged measurement.
- Track seen/rejected versions separately from accepted versions, so an outlier
  is not retried on every UI refresh.
- Use actual source quote timestamps to enforce the existing ten-second quote
  freshness bound; do not refresh an old option quote's age merely because a
  fallback reference changed.

**Reference-only policy:** a measurement version changes only when the source
`(bid_premium_btc, ask_premium_btc)` pair changes. Timestamp/amount-only
heartbeats and fallback-reference revaluations refresh the cached raw row and
normalization metadata, not premium identity. An already-consumed pair cannot
be assimilated again just because its IV or references changed. A pending,
never-consumed version may become usable once when references/clock freshness
permit. A -> B -> A premium changes are distinct versions, whereas call/put
selection changes do not replay a consumed version. Rejected measurement
versions are also marked seen. UI snapshots and periodic ticks never assimilate
pending versions; only feed events can do so.

A large reference/context move can still require the documented overhang reset
and loose-covariance bootstrap. That is an explicit, visible reset, not repeated
measurement precision credited to unchanged premiums.

## Prediction and process noise

After transporting the prior to the current context, use a random walk. Do not
add momentum/velocity states or mean reversion in this version.

```text
v_predicted = transported_v
P_predicted = transported_P + Q(dt)
Q(dt) = q * dt * (0.9 * C + 0.1 * I)
C[i, j] = exp(-abs(z_i - z_j) / correlation_length)
q = process_iv_rate^2
```

`dt` is elapsed time in seconds. The exponential kernel correlates neighboring
knot moves, while the independent component allows local changes. Require
finite nonnegative process noise and positive correlation length.

**Clock convention:** use monotonic elapsed receipt time for prediction
and gap timers; use exchange timestamps for quote freshness and expiry metadata.
Reject or ignore older source versions rather than running prediction backward.
Duplicate/UI events must not artificially advance the filter repeatedly over
the same elapsed interval.

## Coordinate transport and bounded extension

A knot's physical strike changes when `F`, `T`, or `sigma_atm` changes. Carrying
old knot values unchanged would implicitly preserve standardized shape rather
than the previous strike-space curve.

For new knot `z_j`, find its location in the old coordinate system:

```text
K_new_j = exp(log(F_new) + z_j * scale_new)
u_j = (log(K_new_j) - log(F_old)) / scale_old
```

Inside the old domain, evaluate the old natural cubic at `u_j`. This is a linear
map of the old ordinates. Transport covariance with the Jacobian `J` of the
actual transport map:

```text
transported_v = transport(old_v)
transported_P = J * old_P * J.T + W_extension
```

Outside the old domain:

1. Extend from the nearest endpoint using its first derivative, not its cubic
   end segment.
2. Limit the slope to `[-0.10, +0.10]` decimal IV per standardized unit.
3. Allow at most `0.25` standardized units of overhang. If any new knot exceeds
   this bound, reinitialize from fresh quotes rather than extrapolating farther.
4. Add independent extension variance:

```text
d_j = distance beyond the old endpoint, or 0 for an interior knot
extension_sd_j = 0.01 * d_j / 0.25
W_extension = diag(extension_sd_j^2)
```

The maximum allowed extension adds **1 IV percentage point of standard
deviation**, independent of the transported covariance.

**Numerical detail:** for a slope-limited branch, differentiate the
limited map, not the unrestricted cubic extension. If the slope is strictly
clamped, its contribution has zero derivative with respect to the old state;
the endpoint value still contributes. At an exact clamp boundary, use the
unclamped one-sided Jacobian conservatively and test the behavior. This bounded
transport is a first-order covariance approximation, not an exact Gaussian
transformation at the clipping boundary.

## Initialization and curvature regularization

Initialize when at least two distinct fresh in-range strikes are available.
Use the existing regularized static fit on the selected observations, retaining
its default `smoothing=0.01`.

```text
v_initial = fitted nine knot IVs
P_initial = diag(0.05^2)  # independent, loose initial covariance
```

The per-knot initial standard deviation is **5 IV percentage points**. This is a
loose initialization assumption, not a calibrated posterior of the static fit.
Mark bootstrap measurements consumed; do not immediately run a second Kalman
measurement update on those same versions.

Keep the regularized static fit as the reference curve. Do not repeatedly append
curvature pseudo-measurements to the Kalman filter: they could progressively
flatten the smile and inflate apparent certainty. Subsequent stabilization comes
from the temporal prior and the correlated process model.

If a reset is required but bootstrap data is insufficient, display an explicit
waiting/stale state. Do not fabricate missing quote observations or reduce the
nine-knot grid.

## Kalman update and numerical solver

For an eligible scalar observation or coherent batch, calculate:

```text
S = H * P_predicted * H.T + R
solve S * K.T = H * P_predicted
v_updated = v_predicted + K * (y - H * v_predicted)
A = I - K * H
P_updated = A * P_predicted * A.T + K * R * K.T
```

Use `scipy.linalg.cho_factor` and `cho_solve`. Do not form `inverse(S)`. Use the
Joseph covariance update above, then remove numerical asymmetry with
`(P + P.T) / 2`. Validate dimensions, finiteness, covariance symmetry, and
nonnegative variances. The covariance symmetry/eigenvalue tolerance is
`1e-12 * max(1, max(abs(P)))`; measurement variances must be strictly positive.

**Failure behavior:** a failed measurement factorization leaves the
valid predicted state intact and raises a visible diagnostic. Do not silently
fall back to an inverse, pseudoinverse, or flexible zero-pivot solve. An invalid
prior/transport requires explicit recovery or reinitialization, not publication
of non-finite coefficients. Do not disguise model errors with unlimited jitter
injection.

A future Rust port could use the existing Cholesky helpers, but must address their
panic-based failure handling. That port is out of scope here.

## Outliers, regime changes, and gaps

### Innovation gate

For one observation, compare its residual with its predicted innovation standard
deviation: `abs(innovation) / sqrt(S_ii)`. Gate values greater than **5**. Count
only eligible in-range quotes, not stale/invalid/outside observations.

Retain rejected innovations for diagnostics and a rolling three-second regime
window. Same-direction gate breaches at **three distinct strikes** within that
window form a regime-change candidate. Reinitialize from the current fresh quote
set and publish the reason; bootstrap-version tracking prevents immediate double
assimilation. Repeated quotes at one strike cannot satisfy this rule.

This first rule targets broad level jumps. Opposite-direction skew shifts may
not trigger it; include a skew-shift replay and expose that limitation rather
than claiming universal regime detection.

### Missing measurements and feed lifecycle

- Preserve state and predict through short gaps; process uncertainty grows.
- Label stale after **10 seconds without an accepted measurement**. Display source
  quote ages as well as the last accepted-update age.
- After a **60-second gap**, require reinitialization from fresh observations.
- A longer gap is a reset condition even if no periodic prediction ran while
  disconnected. Do not replay historical queued measurements as current data.
- Preserve in-memory states across a short feed toggle/reconnect, subject to the
  same gap rules. Server restart discards them.
- Retire expired/inactive expiry states to keep memory bounded.

## Positivity, output, and controls

Keep the internal linear-filter state unchanged if it estimates negative IVs.
Floor the displayed volatility curve at zero and show a visible clipping warning.
Do not silently project state ordinates onto nonnegative values while leaving
covariance untouched. The initial static fitter's existing nonnegative knot
flooring remains part of bootstrap, not a repeated posterior projection.

Show:

- observed quote points and the current regularized reference curve;
- the filtered natural-cubic curve within observed support;
- nine filtered knot values and derived segment coefficients;
- source-quote residuals/innovations, selected spread, and accepted/gated status;
- quote/update age, warm-up/stale/reset status, clipping and numerical warnings.

No calibrated confidence bands or trading interpretation are included.

Expose two globally shared numeric controls in IV percentage-point units:

| Control | Default | Meaning |
| --- | --- | --- |
| Process IV rate | 0.1 point per sqrt(second) | Raising it generally increases responsiveness |
| Measurement SD at reference spread | 0.5 point at a 1-point spread | Raising it reduces trust in observations |

Convert percentage-point display units to decimal IV internally. Apply changes
from each expiry's next prediction/measurement update without discarding state
or replaying old observations. Do not recompute historical covariance under the
new parameters. Provide an explicit **Reset filters** action.

Other initial constants are:

| Setting | Default |
| --- | --- |
| IV spread floor | 0.1 percentage point = 0.001 decimal |
| Reference IV spread | 1 percentage point = 0.01 decimal |
| Process correlation length | 1 standardized unit |
| Independent process-noise fraction | 0.10 |
| Initial per-knot SD | 5 percentage points = 0.05 decimal |
| Maximum transport overhang | 0.25 standardized units |
| Maximum absolute extension slope | 0.10 decimal IV per standardized unit |
| Additional SD at maximum overhang | 1 percentage point = 0.01 decimal |
| Innovation gate | 5 predicted standard deviations |
| Regime-change window | 3 seconds; 3 distinct strikes; same direction |
| Stale/reset gaps | 10 / 60 seconds |
| Plot refresh | 3 seconds, independent of measurement cadence |

## Implementation boundaries

Keep filter mathematics and per-expiry state management in a testable module
under `example/btc-option-iv/btc_option_iv/`, separate from marimo cells. Use
SciPy's natural `CubicSpline` basis and derivatives for the linear observation
and transport maps; retain the Rust `CubicSmileSection` for bootstrap/reference.
Cross-check their natural-cubic conventions in tests.

Feed the manager from the live feed/controller. Store source identity, the
selected quote set, and normalization context explicitly; they must not depend
on whether a reader cell is visible. Marimo receives immutable view snapshots
and emits configuration/reset requests. The existing surface remains unchanged
in this version: this feature filters expiry smiles, not a joint surface.

The prototype modules are `btc_option_iv/kalman.py` (math/expiry manager),
`live.py` (feed controller), and `views.py` (pure tables/charts). No Rust/binding
API or persistent-state changes are part of this Kalman implementation.
Connected publication predicts every three seconds. Switching the feed off
freezes the detached displayed view explicitly; elapsed-time/gap handling runs
on resume. Short-gap plots may retain last observed support with a stale label.

## Validation and acceptance

Use deterministic offline synthetic streams first, plus existing example
fixtures. Define seeds, timestamps, quote cadence, spreads, truth curves, and
measurement selection before measuring improvements.

Required tests cover:

1. Scalar/matrix updates against independent closed-form calculations; Joseph
   covariance symmetry and numerical positive-semidefiniteness.
2. Exact nine knots and eight segments with two or seven bootstrap quotes;
   one-quote updates after initialization; no fabricated observations.
3. Inverse-spread precision, zero-spread floor, IV-unit conversions, invalid
   values, and call/put selection/version tracking.
4. Duplicate/UI events and bootstrap versions do not reduce covariance again.
5. Forward/ATM/time changes transport a known strike-space curve and its
   covariance; unchanged-context transport is identity.
6. Bounded extension, slope-clamp Jacobians, added uncertainty, overhang reset,
   and insufficient-data recovery.
7. Isolated outliers, same-direction multi-strike level shifts, skew shifts,
   stale labeling, short/long gaps, reconnects, and expiry retirement.
8. Display-only clipping, numerical failures, global control changes, manual
   resets, expiry switching, and immutable UI snapshots.

For a stationary truth scenario, target at least **30% lower RMS frame-to-frame
IV variation** than the regularized reference. Compare the same observation set,
physical strikes, and three-second sampling schedule; exclude bootstrap and do
not count moving coordinates or clipping as jitter improvement.

For a sustained level-step scenario, target **90% response within 10 seconds** of
the observable step. Measure against known truth at fixed physical strikes,
including the outlier/regime policy. Include an isolated-spike scenario to ensure
rapid level-step recovery is not obtained by accepting every bad quote.

These targets apply to specified test scenarios, not all live market conditions.
Report both jitter and lag, all fixture/default settings, and limitations. Do
not lower thresholds to conceal a failed design; tune explicit experimental
parameters or return to the design discussion. Live overlay review supplements,
but does not replace, deterministic tests.

## Confirmation and implementation validation

The user approved the proposed design and requested implementation. The clock,
tie-break, bootstrap covariance, solver-failure, slope-boundary, and module rules
above are the implemented choices. Reference-only behavior is explicit in the
freshness section rather than treating recalculated IVs as independent evidence.

Validation on 2026-09-30:

- 57 offline example tests and 6 native cubic-smile Python tests passed (63 total).
  The initial manager tests were added before implementation and failed because
  the module did not exist. Replay/controller/display tests were added during
  integration; not every test is claimed to have been test-first.
- Seed 430 replay: seven quotes/second, independent `0.005` decimal-IV noise,
  `0.01` IV spreads, fixed physical strikes/F/T, defaults unchanged. Quote-derived
  ATM normalization and actual transport still run. At three-second sampling
  after 30 seconds of warm-up, jitter RMS is `0.001576` filtered vs `0.004489`
  reference: **64.9% reduction**, exceeding the 30% scenario target.
- A sustained five-point level step reaches 90% at the first post-step sample:
  10 ms of simulated receipt time after the seven-quote batch/regime reset,
  within the ten-second target. This is a synthetic batch result, not live
  ten-millisecond latency or a guarantee for every regime/skew move.
- A dropdown regression reproduced quote refreshes overwriting the chosen expiry.
  Separate catalog/selection state now preserves the choice across snapshots and
  catalog additions, falls back when it disappears, and routes to the selected
  view. Chart zoom resets only on expiry change, not every quote refresh.
- An offline controller stream verifies event-time updates before publication,
  reference-only changes, unchanged amounts/timestamps, older events, and
  cancellation without requiring the UI or network.
- Ruff checks/formatting and headless marimo HTML export pass. `marimo check`
  still reports the existing editable-package `self-import` ambiguity and
  Markdown layout diagnostics; its exit code alone is not a clean lint claim.
- A 15-second public-feed smoke initialized all 12 detected expiry filters from
  792 evaluated option rows, with 740 accepted measurement updates and no
  numerical-update warnings. This checks wiring, not calibration or live jitter.

The synthetic fixture details and visualization limitations are also recorded
in the example README. No live confidence-band or arbitrage-free claim is made.

# eSSVI Rust port plan

**Status:** Implemented. The evaluator and sequential calibrator live under `termstructures/volatility`. Acceptance tests cover the in-scope C++ checks except CSV helpers, `runtime_seconds`, and millisecond budgets. A local `-O2` run of the C++ scenario binary misses the summed-loss, RMS, and fallback-rate ceilings, so the Rust scenario test locks convergence, validation, the dense grid, and the cold-search budget instead of those ceilings.
**Map:** [eSSVI Rust port](../.wayfinder/maps/essvi-rust-port.md)
**Oracle:** `~/git/ql-essvi`, in particular `cpp/essvi_surface.cpp`, `cpp/calibrator.cpp`, `docs/parameterization.md`, and the in-scope tests named below.
**Home:** `crates/libitofin/src/termstructures/volatility/`

Where this plan and the C++ disagree about control flow, constants, or numeric branches, the C++ wins. This plan records the public shape, the required deviations, and the acceptance bar.

## 1. Decisions

| Decision | Outcome |
| --- | --- |
| Public types | Standalone `Slice`, `Surface`, `Quote`, `WarmOptions`, and `CalibrationResult`. No `SmileSection` or `BlackVolTermStructure`. |
| Failure | A fit that does not converge still returns `CalibrationResult`. |
| Quotes | In-memory quotes only. No `readQuotes`, `writeParameters`, or `writeDiagnostics`. |
| Result floats | Include the band-loss objective, excess ratios, warm and reference loss, loss scale, and loss tolerance. |
| Algorithm | Transliterate the rho-window search, psi bounds, calendar-headroom retry, and warm-update state machine. |
| Schedule | Sequential rho search. Same windows. Same total bound-call count. No worker pool. |
| Acceptance | Exact slice values, path, ages, solver counts, covariance, and quote-quality assertions from the in-scope tests. |
| Left out of acceptance | Scenario millisecond budgets. CSV helper assertions. |

## 2. Modules

- `essvi.rs` — slice formula, Corbetta interpolation, Martini–Mingone bound, butterfly and calendar checks, `validate`.
- `essvi_calibrator.rs` — `calibrate`, `calibrate_warm`, and the result types.
- Register both from `termstructures/volatility/mod.rs` and re-export the public types.

Test-only helpers that the C++ tests call stay available to Rust tests: `mmPsi2BoundReference`, `mmPsi2BoundSearchHint`, and `passesAdaptiveWeightedQuoteCap`. They are not part of the public surface.

## 3. Evaluator

`Slice` is `(expiry, theta, psi, rho)`. `k = ln(strike / forward)`.

```text
w(k) = 1/2 * (theta + rho*psi*k
              + sqrt((psi*k + theta*rho)^2 + theta^2*(1-rho^2)))
```

`implied_volatility` is `sqrt(w / expiry)` for `expiry > 0` and `w >= 0`, and non-finite otherwise.

`slice_at` clamps outside the first and last listed expiries. Between listed slices it linearly interpolates `(theta, psi, rho*psi)` and recovers `rho` from the interpolated product. An empty surface returns a zero slice from `slice_at` and a non-finite variance from `total_variance`.

`mm_psi2_bound` follows `boundEstimate` in `cpp/essvi_surface.cpp`, including the 257-point rho workspace, the refinement, and the per-call counter. The search hint follows `mmPsi2BoundSearchHint`. The reference used by the million-sample test follows `mmPsi2BoundReference`.

`butterfly_safe` requires `theta > safety`, `psi > safety`, `|rho| < 1 - safety`, the Roger-Lee bound `psi*(1+|rho|) < 4 - safety`, and `psi^2 <= mm_psi2_bound - safety`. Default safety is `1e-10` on the public function, matching the C++ header.

`calendar_safe` requires a later expiry, `theta` and `psi` strictly increasing by `safety`, and `psi` strictly inside the Hendriks–Martini / Corbetta floor and the Mingone upper bound `psi_1 / theta_1 * theta_2`.

`validate` returns an empty string for a surface that passes butterfly on every slice and calendar on every adjacent pair. The failure strings match `validate` in `cpp/essvi_surface.cpp`.

## 4. Calibration API

`Quote` carries `id`, `expiry`, `forward`, `discount_factor`, `strike`, `bid_iv`, `ask_iv`, `mid_iv`, `included`, and `weight`. The default weight is `1`.

`WarmOptions` carries `quote_tol` (default `1e-5`), `loss_tol` (default `0.05`), and `quote_cap` (default `0.1`). `loss_tol` remains because warm results echo it as `loss_tolerance` and the warm tests set it.

`CalibrationPath` is `cold`, `warm`, `fallback`, or `degraded`.

`CalibrationResult` keeps:

- `surface`, including partial slices when a cold fit fails
- `converged`, `error`, `objective`
- `path`, `reuse_ages`, `degraded_ages`
- posterior covariance as six `f64` values `[00, 01, 02, 11, 12, 22]` in `(log theta, atanh rho, log psi)`, one row per slice
- solver counts: `iterations`, `objective_evaluations`, `local_solver_slices`, `local_solver_starts`, `local_objective_evaluations`, `local_mm_bound_calls`, `mm_bound_calls`, `restricted_cold_searches`, `full_cold_searches`, `forced_full_cold_searches`, `degraded_slices`, `slices_refit`, `slices_reused`, `fallback_slices`, `quality_checks`, `quality_quote_count`, and `fallback_reason_counts`
- quote quality: `quality_maximum_excess_ratio`, `quality_rms_excess_ratio`, `worst_slice_max_excess_ratio`, `worst_slice_rms_excess_ratio`, `quality_warm_loss`, `quality_reference_loss`, `quality_loss_scale`, `loss_tolerance`

It omits `timings` and `runtime_seconds`.

These error strings are exact, because tests compare them:

- `included quote weight must be finite and positive`
- `no ATM quote`

Other errors match the C++ text, including the empty-feasible-set and `validate` messages.

## 5. Control flow to transliterate

Follow `cpp/calibrator.cpp` in branch order. Evaluate rho candidates sequentially. Add each candidate's Martini–Mingone calls into the same total the C++ stores in `mm_bound_calls`. Do not spawn the `RhoWorkerPool`.

Load-bearing constants:

| Constant | Value |
| --- | --- |
| `kEps` | `1e-8` |
| `kCalendarHeadroom` | `0.005` |
| full-cold budget | `8` |
| degraded age before a forced full cold | `3` |
| cold state variance | `0.25` on the diagonal |
| process noise | diagonal `(1e-2, 5, 5)`, added once per warm update |
| posterior variance floor | `1e-8` |
| cold rho window | `[-0.95, 0.95]` at step `0.05`, then a `±0.06` refinement at step `0.01` |
| restricted rho window | prior `±0.15`, clamped to `±0.95`, then the same refinement |
| theta multipliers after a cold best | `0.90, 0.95, 1.00, 1.05, 1.10, 1.20` |

Cold `calibrate` groups quotes by expiry and fits slices in expiry order. Each slice tries the strict calendar floor first. If that winner already clears the `0.5%` psi-ratio margin, it is kept. Otherwise the fit retries with `kCalendarHeadroom` and keeps the headroom winner when that retry converges.

`calibrate_warm` keeps the C++ reuse rule, local bounded starts, quote-cap gate, restricted cold fallback, full-cold budget, and degraded path. Covariance follows the generalized-Bayes section of `docs/parameterization.md`: cold slices start at `0.25 I`, a valid exact-expiry warm prior receives the process noise once, and active band residuals add the Gauss–Newton information in the transformed coordinates.

The objective is the weighted squared total-variance residual outside the bid/ask band. A weight of `2` scales that objective by `4`.

## 6. Acceptance tests

Port the assertions in these C++ files into `crates/libitofin` tests:

- `tests/test_essvi.cpp`
- `tests/test_mm_bound.cpp`
- `tests/test_calendar_headroom.cpp`
- `tests/test_calibrator_regression.cpp`
- `tests/test_warm_calibrator.cpp`
- `tests/test_warm_scenario_regression.cpp`

Keep the numeric tolerances those files already use, including the `1e-9` relative match to `tests/baseline_essvi_40x100.csv`. That baseline is a parameter table, not a quote-file API.

Quote generators used only to build those tests (`syntheticChain`, `realisticChain`, `evolveWarmScenario`) live in the Rust test module. The Deribit fixture may be read by the test. The library still has no CSV API.

Drop these assertions when porting the tests:

- `readQuotes` and `writeDiagnostics` behavior in `test_calibrator_regression.cpp`
- `runtime_seconds` is finite
- warm p99, quiet median, and quiet p99 millisecond budgets in `test_warm_scenario_regression.cpp`

Keep the scenario file's convergence, dense-grid, degraded-state, fallback-rate, and quote-quality limits.

## 7. Before coding

This document closes the planning map. Writing the Rust modules and tests needs a separate authorization.

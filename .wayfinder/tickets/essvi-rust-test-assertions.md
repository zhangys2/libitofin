---
title: eSSVI in-scope C++ test assertions
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Grok (research session 2026-11-12)"
map: ../maps/essvi-rust-port.md
---

# eSSVI in-scope C++ test assertions

## Question

Which assertions in the in-scope C++ tests lock evaluator values, butterfly and calendar checks, cold `calibrate`, and `calibrate_warm`, and which of those assertions lock search internals rather than slice parameters or pass/fail?

## Context

Search internals include iteration counts, `CalibrationPath`, posterior covariance entries, reuse and degraded ages, and timing or bound-call counters. The reproduction-strategy ticket needs this split before it can choose a transliteration versus an output match.

Read only:

- `~/git/ql-essvi/tests/test_essvi.cpp`
- `~/git/ql-essvi/tests/test_mm_bound.cpp`
- `~/git/ql-essvi/tests/test_calendar_headroom.cpp`
- `~/git/ql-essvi/tests/test_calibrator_regression.cpp`
- `~/git/ql-essvi/tests/test_warm_calibrator.cpp`
- `~/git/ql-essvi/tests/test_warm_scenario_regression.cpp`

Leave `test_essvi_ql.cpp`, `test_quantlib.cpp`, and the benchmark and Python pipeline tests unread for this ticket. They sit outside the charter. Record the assertion inventory on this ticket. Do not choose the reproduction strategy here.

## Resolution

Resolved 2026-11-12 by reading the six listed files. Assertions use `assert`. Nothing here chooses the reproduction strategy.

### Evaluator, butterfly, and calendar

`test_essvi.cpp` locks surface values and booleans. ATM total variance matches `theta` within `1e-12`. `sliceAt` matches a listed slice within `1e-12` and moves theta between listed expiries. `butterflySafe` accepts a Gatheral–Jacquier-interior slice and a short-dated slice at safety `1e-8`. One slice violates the Gatheral–Jacquier `psi^2` bound while `psi*(1+|rho|) < 4`. `calendarSafe` rejects a rising-phi pair. `validate` is empty on the two-slice example. Cold `calibrate` on a synthetic three-slice chain must converge, pass `validate`, stay inside every bid/ask IV band within `1e-5`, and recover a non-monotone rho order (`s1.rho` above both neighbors, `s0.rho < -0.25`, `s1.rho > -0.25`). No included quotes returns `converged == false` and a non-empty error. This file does not lock solver counts.

`test_mm_bound.cpp` locks the bound value. On one fixed `(theta, rho)` and on 4096 seeded samples, `mmPsi2Bound` is finite and positive, no higher than an independent fine-grid infimum by `1e-12` relative, and no lower by `1e-9` relative. On 1,000,000 seeded samples it stays within `1e-12` relative of `mmPsi2BoundReference`, and `detail::mmPsi2BoundSearchHint` matches that reference within `1e-14` relative. The seed is `0x8f3f73b5cf1c9ad1`.

`test_calendar_headroom.cpp` locks a calibrated surface, not the search counters. The Deribit fixture must converge. `validate` is empty, every slice is `butterflySafe`, total variance on `k` from `-0.5` to `0.5` in steps of `0.01` is positive, variance is calendar-monotone within `1e-8`, and call slopes are convex within `1e-8`. On the fast warm scenario, updates 18, 27, and 31 must keep at least one adjacent pair whose psi clears the strict floor and still sits on or under the `1.005` headroom floor. The fixture is loaded with `readQuotes`; the assertions are on the fit.

### Cold and warm results

`test_calibrator_regression.cpp` locks both outputs and counters. Two cold fits of the 40-by-100 synthetic chain match bitwise on `objective`, `iterations`, `objective_evaluations`, `mm_bound_calls`, and every `(expiry, theta, psi, rho)`. Against `tests/baseline_essvi_40x100.csv`, expiry matches bitwise and theta, psi, and rho match within `1e-9` relative. Weight `2` scales the cold objective by `4` within `1e-12` relative. Invalid included weights, in memory or via CSV, fail cold and warm with the exact error `included quote weight must be finite and positive`. A warm fit must have `path == warm`, `local_solver_starts > 0`, and `objective` equal to the recomputed weighted band loss within `1e-12` relative, or within `1e-14` absolute in the weighted-equation case. Unit and weight-20 slices must differ by more than `1e-10` in theta, psi, or rho. `quality_checks == 1`, `restricted_cold_searches == 1`, `fallback_reason_counts.loss_above_gate > 0`, and `slices_refit > 0`. `detail::passesAdaptiveWeightedQuoteCap` is locked on either side of `0.055` and `0.22`.

The same file also locks CSV behavior the charter left out of the Rust API. `readQuotes` maps a missing weight column and a blank weight to `1`, and a weight column of `2` to `2`. `writeDiagnostics` writes the header `model,converged,objective,iterations,runtime_seconds,error`.

`test_warm_calibrator.cpp` locks the warm state machine, not only the slices. Repeated warm fits match slice parameters bitwise. Several paths require an exact `CalibrationPath` (`warm`, `cold`, `fallback`, `degraded`). Reuse and refit counts are exact (`40/0`, `0/40`, `3/0`, sums to the slice count). `reuse_ages` stay at most `12`. Failed warm fits have an empty slice list and a non-empty error, including the exact error `no ATM quote`. Quote-cap checks bound `quality_maximum_excess_ratio` by `quote_cap` plus `1e-8`, `1e-10`, or `1e-12`, and one path requires `quality_rms_excess_ratio <= 0.02`. Objective stays within `1e-12` of the previous cold objective, or within a `1.05` factor of a fresh cold fit. Posterior covariances are finite, diagonals are at least `1e-8`, the 2×2 and 3×3 principal minors are positive, and several cold-start paths require bitwise covariance equality. Slice parameters are bitwise equal after one calendar-repair path. Rho must move by more than `0.2` from the prior and land within `0.1` of a target on the escape path.

The same file locks exact integers: `local_solver_starts` of `3` and of `1`, `local_solver_slices` of `1` or `2`, `local_objective_evaluations` of `2` and a cap of `25`, `local_mm_bound_calls <= 11`, `quality_checks` of `0` or `1`, `full_cold_searches` of `8` or `10`, `forced_full_cold_searches` of `0`, `1`, or `10`, `fallback_reason_counts.calendar_repair_failed == 1`, and `degraded_ages.front() == 0`. It also requires `mm_bound_calls > 0`, `objective_evaluations >= iterations`, and a finite `runtime_seconds`, without a millisecond budget. `loss_tolerance` must equal `WarmOptions.loss_tol`.

`test_warm_scenario_regression.cpp` locks scenario outcomes for quiet, normal, fast, and front. Warm and cold fits converge, `validate` stays empty, and the dense-grid violation count is `0`. `full_cold_searches` stays within `8` plus the forced count. A degraded path must report matching `degraded_slices` and `degraded_ages`. Against the cold fit, the share above the cap may rise by at most `2` percentage points, the summed band-loss ratio is at most `1.20`, and the RMS excess-IV gap is at most `0.05` points. Fallback updates stay at or below `25%` on fast and `5%` otherwise. This file also locks wall-clock budgets: warm p99 at most `20 ms` on fast and `10 ms` otherwise, and on quiet a median at most `0.47 ms` and a p99 at most `7.6 ms`.

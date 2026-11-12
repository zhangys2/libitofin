# eSSVI Rust port

## Destination

Produce a closed set of porting decisions and a written implementation plan for the native
eSSVI library in `~/git/ql-essvi`: the total-variance evaluator, butterfly and calendar
checks, `calibrate`, and `calibrate_warm`. The plan targets `crates/libitofin` under
`termstructures/volatility`, with the checked-in C++ code, its in-scope tests, and
`docs/parameterization.md` as the behavior oracle.

Complete when every decision ticket is closed and the implementation plan is written.
Rust code waits for a separate authorization after that gate.

## Notes

Charter choices, confirmed 2026-11-12. These are not tickets; do not re-litigate them
without saying so explicitly.

- The map ends at decisions plus an implementation plan. Writing the Rust port is a later
  authorization.
- In scope: evaluator, arbitrage checks, cold `calibrate`, and `calibrate_warm`.
- Home: `crates/libitofin/src/termstructures/volatility`. `~/git/ql-essvi` stays the oracle.
- The oracle is the C++ implementation, its tests, and `docs/parameterization.md`. This map
  does not reopen a paper-by-paper audit. Parameterization choices already live on
  [eSSVI vs CVI benchmark](../../../ql-essvi/.wayfinder/maps/essvi-vs-cvi-benchmark.md).
- Survey facts, 2026-11-12: `essvi::Surface` stores per-expiry `(theta, psi, rho)` and
  evaluates total variance at log-moneyness. Butterfly uses the Martini–Mingone infimum.
  Cold calibration searches rho windows inside psi bounds, with an ATM theta anchor and a
  calendar-headroom retry. Warm update carries a 3×3 covariance in
  `(log theta, atanh rho, log psi)`, plus fallback and degraded paths. libitofin has
  `SmileSection` and no eSSVI type.

Open tickets use the `essvi-rust-` prefix and set `map` to this file. The frontier is any
ticket with status open, nothing in `blocked_by`, and no claim.

Skills: `grilling` for HITL tickets. `domain-modeling` only if a public type name becomes
canonical or a hard-to-reverse trait choice needs an ADR.

## Decisions so far

- [eSSVI public Rust types](../tickets/essvi-rust-public-types.md) — standalone `Slice`, `Surface`, `Quote`, `WarmOptions`, and `CalibrationResult`; no `SmileSection` or `BlackVolTermStructure` in this plan.
- [eSSVI calibration failure result](../tickets/essvi-rust-failure-result.md) — a non-converged fit still returns `CalibrationResult` with partial slices, covariance, path, ages, and solver counts; wall-clock timings are omitted.
- [eSSVI quote input boundary](../tickets/essvi-rust-quote-io.md) — in-memory quotes only; CSV read and write stay in the C++ tree.
- [eSSVI objective and quote-quality fields](../tickets/essvi-rust-quality-fields.md) — `CalibrationResult` also carries the band-loss objective, excess ratios, warm and reference loss, loss scale, and loss tolerance.
- [eSSVI in-scope C++ test assertions](../tickets/essvi-rust-test-assertions.md) — slice values and arbitrage checks are locked, and so are exact warm-path counts, bitwise covariance, and a scenario millisecond budget; CSV helper assertions are present but sit outside the Rust API.
- [eSSVI reproduction strategy](../tickets/essvi-rust-reproduction-strategy.md) — transliterate the rho-window search, psi bounds, calendar-headroom retry, and warm-update state machine; Rust acceptance is the exact slice, path, age, count, covariance, and quote-quality assertions.
- [eSSVI rho-search schedule](../tickets/essvi-rust-rho-search-schedule.md) — sequential rho search with the same windows and total bound-call accounting; the C++ worker pool stays out.
- Implementation plan (2026-11-12) — [eSSVI Rust port plan](../../docs/essvi-rust-port-plan.md). Decision tickets are closed. Rust code still waits for a separate authorization.

## Not yet specified

None. The implementation plan is [eSSVI Rust port plan](../../docs/essvi-rust-port-plan.md).

## Out of scope

- Rust implementation in this map — the charter keeps code behind a separate authorization.
- CVI benchmark harness, CSV scoring scripts, and the QuantLib C++ adapter (`cpp/essvi_ql.*`)
  — left in `ql-essvi`.
- A fresh audit of Gatheral–Jacquier, Corbetta, Hendriks–Martini, or Mingone — the C++
  oracle and the existing benchmark map already record those choices.
- Python, C-FFI, and Go bindings — this destination is the Rust library plan.
- `SmileSection` and `BlackVolTermStructure` adapters — [public types](../tickets/essvi-rust-public-types.md) left them out of this plan.
- CSV quote, parameter, and diagnostics file helpers — [quote input](../tickets/essvi-rust-quote-io.md) left them in the C++ tree.
- Wall-clock calibration timings — [failure result](../tickets/essvi-rust-failure-result.md) omitted them from `CalibrationResult`.
- Scenario millisecond budgets, and CSV helper assertions as Rust acceptance tests — [reproduction strategy](../tickets/essvi-rust-reproduction-strategy.md) left both out of the acceptance bar.
- The C++ rho-search worker pool — [rho-search schedule](../tickets/essvi-rust-rho-search-schedule.md) keeps the search sequential.

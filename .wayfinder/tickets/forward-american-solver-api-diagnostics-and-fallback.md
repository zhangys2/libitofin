---
title: American Solver API, Diagnostics & Fallback
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Claude + Yuan (grilling session 2026-10-26)"
map: ../maps/implied-forward-from-options.md
resolution: "Single Rust entry point with a nested AmericanModelConfig; model path runs for AmericanEquity whenever discount_factor is given. Orthogonal result fields (tier, flags bitset, solved carry, call/put IV gap, iterations, accurate prices used); ForwardStatus gains only WarningImplausibleCarry (amended 2026-10-27: also WarningModelFallback, and a model_failure cause; see Amendments). Optional atm_vol exposed everywhere, estimated from quotes when absent. Deterministic work budget (max_accurate_prices), no clock in the core. Correction at mid, applied to both sides, checked against per-side in fixtures. FFI: additive _with_options function and extended result, ABI stays 1. Python: extend implied_forward. Go: ImpliedForwardWithOptions plus typed ForwardConvention; existing Go status/convention defects fixed first."
---

# American Solver API, Diagnostics & Fallback

## Question

What changes to the public surface of `ImpliedForward` support the model-based path: new `ImpliedForwardConfig` fields (pricer choice, tolerance, iteration cap, dividend schedule), a change to `MarketConvention::AmericanEquity`, new `ForwardDiagnostics` fields (iterations, final IV gap, EEP magnitude), and new `ForwardStatus` values (for example non-convergence)? When the iteration fails to converge or the pricer errors, does the result fall back to the corridor estimate, and how is that flagged?

Also decide how the change propagates through [C-FFI, PyO3 Python Bindings, and Downstream Feed Migration](forward-ffi-python-and-feed-migration.md) without breaking existing callers.

## Context

- Existing types: [Rust Core API, Structs, and Trait Representation](forward-rust-api-and-type-system.md) and [Arbitrage Diagnostics & Crossed-Quote Reporting](forward-arbitrage-diagnostics-and-reporting.md).
- Method settled in [De-Americanization Method & Tier Structure](forward-deamericanization-method-and-tiering.md): default model-based tier, corridor fallback flagged by tier, Fast/Accurate config enum with automatic fall back to Fast on a latency overrun, call/put IV gap reported as a diagnostic. Inputs settled in [American Carry, Dividend & Rate Inputs](forward-american-carry-dividend-and-rate-inputs.md): required `discount_factor` (else corridor tier), optional dividend schedule used at the Accurate level, carry search and flag ranges as config fields, a schedule makes Accurate the default level. Open here: the names and representation of the tier, schedule-ignored and implausible-carry flags; where the ATM vol for the widened corridor comes from. The pricer question is answered, see [American Pricer Inventory, Accuracy & Latency Budget](forward-american-pricer-inventory-and-latency.md).

## Resolution

Resolved 2026-10-26 via grilling rounds 1 and 2 (ten questions; every answer matched the recommendation).

### Facts found while reading the current surface

- `ForwardStatus` is a single enum, so only one signal can be reported at a time. The new design needs tier, schedule-ignored, latency fallback and implausible carry independently.
- The FFI and Python entry points hard-code `atm_vol: None`, so the American corridor always falls back to the built-in 0.30 vol (`unwrap_or(0.30)` in `implied_forward.rs`).
- Repo FFI convention for extended functions: a `const ItofinXOptions*` `repr(C)` struct with `has_*` int32 flags and a `…_options_default` helper (see `itofin_piecewise_curve_new_with_options`). `itofin_abi_version()` is 1 and the Go session init checks `== 1`.
- Go defects in `sdk/go/forward.go`, untested because `forward_test.go` only covers `convention` 0:
  - The doc says convention 1 is American and 2 is Crypto; the FFI uses 0 European, 1 CryptoInverse, 2 AmericanEquity.
  - Go maps status codes 1/2/3 to InsufficientPairs/ArbitrageViolations/CrossedMarket and everything else to FailedFit, while the FFI emits 0 Valid, 1 CrossedSyntheticQuotes, 2 BoxSpreadArbitrage, 3 ImplausibleDiscountFactor, 4 SpotDeviationExceeded, 5 DegradedAmericanBounds. `IsCrossed` is set for code 3, which is the discount-factor warning.
  - `MinSpread`, `MaxSpread` and `BoxArbitrageViolations` are hard-coded to 0, and `ImpliedCarryRate` is non-nil even when the FFI sends NaN.

### Rust

1. **Surface.** `ImpliedForward::calculate` stays the single entry point. `ImpliedForwardConfig` gains a nested `AmericanModelConfig` (with `Default`): level (`Auto`, `Fast`, `Accurate`), dividend schedule as `Vec<(Time, Real)>` pairs of year-fraction time and amount, carry search and flag ranges, correction-pass cap and tolerance, work budget, optional `atm_vol`. For `AmericanEquity` the model-based path runs whenever `discount_factor` is supplied; American callers that already pass one now get the model-based result, labeled by tier. Invalid inputs (schedule time outside $(0, \text{expiry}]$, negative amounts) are hard errors, not flags.
2. **Result and flags.** New orthogonal fields: `tier` (`Corridor`, `ModelFast`, `ModelAccurate`), a `flags` bitset (`ScheduleIgnored`, `LatencyFallback`, `NonConverged`), and numeric diagnostics `solved_carry`, `call_put_iv_gap`, `iterations`, `accurate_prices_used`. `ForwardStatus` gains only `WarningImplausibleCarry`.
3. **ATM vol.** An optional `atm_vol` is exposed in every layer. When absent it is estimated from the quotes (the model path already computes per-quote IVs) and falls back to 0.30 only when estimation is impossible, for example in the corridor tier without a discount factor.
4. **Latency budget.** A deterministic `max_accurate_prices`, defaulted from the corrector's cost (about 2 ms per CRR 400 price). On overrun the correction covers the strikes closest to ATM first, then falls back to Fast with `LatencyFallback`. No wall clock inside the core.
5. **Correction side.** The premium correction is computed at mid and applied to both bid and ask. The implementation's fixtures must compare it with a per-side computation; a per-side flag can be added later without changing the shape.

### FFI, Python, Go

6. **FFI.** Add `itofin_implied_forward_options_default` and `itofin_implied_forward_calculate_with_options`, taking the existing arguments plus an `ItofinImpliedForwardOptions` (`has_*` flags; dividend arrays and count; level code; carry ranges; work budget; tolerance; atm vol) and writing an `ItofinImpliedForwardResultEx` (the old fields plus tier, flags, solved carry, call/put IV gap, iterations, accurate prices used). The existing function and struct are untouched and the ABI version stays 1. The old function runs the model path at default settings for American callers with a discount factor and does not report the tier. The status code for `WarningImplausibleCarry` is 6.
7. **Python.** Extend `implied_forward(...)` with keyword arguments `atm_vol`, `dividends` (list of `(time_years, amount)`), `level` (`"auto"`, `"fast"`, `"accurate"`), the carry search and flag ranges, `max_accurate_prices`. The result gains `tier` (`"corridor"`, `"model_fast"`, `"model_accurate"`), `flags` (`list[str]`), `solved_carry`, `call_put_iv_gap`, `iterations`, `accurate_prices_used`, all `None` for non-American results.
8. **Go surface.** Add `ImpliedForwardWithOptions(pairs, expiryYears, convention, spot, discountFactor, opts *ImpliedForwardOptions)` returning an extended result. `ImpliedForwardOptions` follows the SDK pattern of pointer fields for optional values. The existing `ImpliedForward` keeps its signature.
9. **Go fixes, first step of the delivery.** Correct the status mapping, `IsCrossed`, the convention doc comment and the nil/NaN carry handling; add Go tests for every status code and the American convention; note the change in the release notes, since Go callers currently see wrong status strings.
10. **Go constants.** A typed `ForwardConvention` with `ForwardEuropeanVanilla = 0`, `ForwardCryptoInverse = 1`, `ForwardAmericanEquity = 2`. The new function takes it; the old function keeps `int32` with the comment corrected to the real codes.

### Settled during delivery

The corridor width (`corridor_kappa`) stayed at 1.0 and the validation fixtures and the per-side check of item 5 were built; see the map's delivery entry.

## Amendments

Decided during the review of the delivery pull requests (2026-10-26 and 2026-10-27). Each supersedes the item it names.

- **Item 2, status.** `ForwardStatus` also gains `WarningModelFallback` (FFI and Go code 7, Python `"WarningModelFallback"`), reported when the model path failed and the clean corridor estimate is returned in its place. A model failure that read as `Valid` was the reviewers' main objection: the old ABI and Go `ImpliedForward` cannot see the `NON_CONVERGED` flag, so the status had to carry it. A corridor fit that already has a warning keeps that warning, and an Accurate-stage failure that falls back to Fast stays `Valid`, because that is still a model result (the flag says it happened). (#279)
- **Item 2, flags.** A fourth flag, `PairsDropped` (value 8, `"pairs_dropped"`), is set when more than a quarter of a denominator cannot be de-Americanized. On the Fast path the denominator is the strike window passed into `solve` (`pairs.len()`). On the Accurate path it is the budgeted correction subset (`attempted` in `accurate_stage`, `corrected.len()`); `deamericanize` runs only on that subset. Strikes the budget never corrects do not enter the fraction. `pairs_pruned` is `total_received.saturating_sub(solution.pairs.len())` in `calculate_american_model`: every quote left out of the fit, including invalid quotes, the corridor cut, and the `max_pairs` cap.
- **Item 2, cause.** New `ModelFailure` (`#[non_exhaustive]`), set on `ImpliedForwardResult.model_failure` with `NON_CONVERGED`, both for a corridor fallback and for an Accurate-to-Fast fallback: insufficient pairs, dividends above spot, pricer unavailable, no implied vol, carry not bracketed, carry solve failed, too few strikes, parity fit failed. FFI: `model_failure: u32` appended to `ItofinImpliedForwardResultEx` (0 none, 1 to 8 in that order). Python: `model_failure: str | None` (snake-case names). Go: `ImpliedForwardResultEx.ModelFailure`. Invalid inputs, such as a non-positive discount factor, stay hard errors of the call. When the Fast fallback fails too, the Accurate stage's cause is reported.
- **Spot gate.** The model path does not apply `max_spot_deviation`: the forward of an equity follows the rate and the carry, so the gate rejected a correct long forward. Plausibility is judged by `carry_flag_range` per year.
- **Carry search with a schedule.** The search range is a range of the residual carry, so with a dividend schedule it is shifted by `ln(S/(S-PV))/T` before it is applied to the flat yield. Dividends that land on the same virtual day in the pricer are added, not overwritten.
- **Carry solve.** The refinement runs on the strikes that invert at both ends of the bracket (halving the bracket until at least two and half of them do) and fails with `carry solve failed` instead of following a NaN or a changing strike set. The reported call/put gap is the objective's own value at the solution.
- **Go step 9.** Done first, as decided: status mapping, `IsCrossed` (computed from the strict bounds, since a warning raised earlier hides the crossed status but not the crossed book), convention comment, nil carry only for NaN, tests for every status code.

### Still open

- `#[non_exhaustive]` on `ForwardStatus` and the config structs: not applied, since it would break struct-literal construction (`..Default::default()`) in downstream code and the bindings. Needs a decision together with the changelog and version.
- The pricer's error text is not carried through; `ModelFailure` is the contract.
- The bracket search can accept a mean gap within 1e-9 before the strike set is locked; the window is tiny.
- Legacy Go `ImpliedForward` still hardcodes `MinSpread`, `MaxSpread`, and `BoxArbitrageViolations` to 0. `ImpliedForwardWithOptions` fills those three from the extended result. Step 9 corrected status mapping, `IsCrossed`, the convention comment, and nil carry on NaN, and left the three hardcoded fields as they were.

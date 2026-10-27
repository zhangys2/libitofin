# Implied Forward Price & Discount Factor Engine

## Destination

Design and implement a unified implied forward price and discount factor estimation engine (`ImpliedForward`) in `libitofin` that calculates synthetic forward bid-ask intervals $[F_{\text{bid}}, F_{\text{ask}}]$, weighted point estimates $F^*$, and arbitrage diagnostics from option quotes with bid-ask spreads across European, American (model-free ATM strike filtering as the fast tier, plus model-based de-Americanization via iterative early-exercise-premium subtraction and/or call/put IV matching), and crypto-inverse markets. The engine must be implemented in Rust core (`crates/libitofin`), exposed via C-FFI (`crates/libitofin-ffi`) and Python (`crates/itofin-py`), validated against analytical fixtures and synthetic arbitrage scenarios, and integrated into the live SPX and BTC feeds in `example/btc-option-iv`.

## Notes

- **Domain**: Equity index (SPX), single stock/ETF (American options), cryptocurrency options (Deribit inverse BTC/ETH, cash-settled linear).
- **Core Principles**:
  - European Put-Call Parity: $C(K) - P(K) = D(T) \cdot (F - K)$.
  - Conversion / Reversal Arbitrage Bounds: $F_{\text{bid}} = K + (C_{\text{bid}} - P_{\text{ask}})/D$, $F_{\text{ask}} = K + (C_{\text{ask}} - P_{\text{bid}})/D$.
  - Weighted Least Squares (WLS): Weights $w_i = 1 / (\Delta C_i^2 + \Delta P_i^2)$ to suppress illiquid wing noise.
  - Model-Free American Bounds: Narrow-band ATM strike restriction where early exercise premium is negligible, plus analytical Merton bounds.
  - Model-Based De-Americanization (reopened 2026-10-25): American parity is an inequality, $C_A - P_A = (C_E - P_E) + (\text{EEP}_C - \text{EEP}_P)$, so the corridor filter leaves a rate-, dividend- and maturity-dependent bias. Two candidate fixes, method still to be chosen: (1) iterative EEP subtraction then WLS parity regression, repeated to convergence; (2) solve the single unknown $F$ (or borrow $q$) so the American call and put implied vols agree at each strike, $\text{IV}_C(K) = \text{IV}_P(K)$, with $D$ fixed from an OIS curve.
  - Inverse Crypto Conventions: Deribit coin-margined payoffs where effective discount is $D = S / F$ and premiums are quoted in BTC.
- **Related Modules**:
  - Core Rust: [`crates/libitofin/src/termstructures/`](../../crates/libitofin/src/termstructures/)
  - Python bindings: [`crates/itofin-py/`](../../crates/itofin-py/)
  - C-FFI: [`crates/libitofin-ffi/`](../../crates/libitofin-ffi/)
  - Existing Python feeds:
    - [`example/btc-option-iv/spx_option_iv/feed.py`](../../example/btc-option-iv/spx_option_iv/feed.py) (`parity_forward`)
    - [`example/btc-option-iv/btc_option_iv/market.py`](../../example/btc-option-iv/btc_option_iv/market.py) (`forward_from_future_quote`, `forward_data`)
- **Skills**: `grilling` for HITL design alignment, `wayfinder` for durable decision tickets, `tdd` for test-driven delivery.

## Decisions so far

- **Destination & Scope Confirmation (2026-10-07 via Grilling Round 1)**: Unified Rust engine + C-FFI + `itofin-py` bindings + downstream refactoring of `spx_option_iv` and `btc_option_iv` to consume the engine.
- **Market & Exercise Scope (2026-10-07 via Grilling Round 1; American tier amended 2026-10-25, see frontier)**: Tiered model-free approach — exact parity & WLS for European and Inverse Crypto; strike-restricted parity (filtering ITM strikes where early exercise is non-zero) + analytical early exercise bounds $[F_{\min}, F_{\max}]$ for American options.
- **Bid-Ask Spread Modeling (2026-10-07 via Grilling Round 1)**: Dual contract providing both synthetic forward spread $[F_{\text{bid}}, F_{\text{ask}}]$ from conversion/reversal bounds and weighted point estimate $F^*$ via WLS / robust median with weights $1/(\Delta C_i^2 + \Delta P_i^2)$.
- **Discount Factor Handling (2026-10-07 via Grilling Round 1)**: Hybrid mode — solve for $F$ with fixed $D$ if discount factor/curve provided; jointly solve $(F, D)$ via cross-strike regression if omitted.
- **Architectural Placement (2026-10-07 via Grilling Round 1)**: Core Rust in `crates/libitofin`, bound in `libitofin-ffi` and `crates/itofin-py`, consumed by example feeds.
- [Implied Forward Mathematical Formulation & WLS Objective](../tickets/forward-math-wls-and-bounds-formulation.md) — Closed-form WLS regression with spread weights $w_i = 1 / (\Delta C_i^2 + \Delta P_i^2 + \epsilon^2)$ estimating $(F^*, D^*)$ simultaneously via $y = D \cdot F - D \cdot K$. Strict + robust synthetic bounds $[F_{\text{bid}}, F_{\text{ask}}]$ per strike.
- [Inverse Crypto Option Parity & Cash Flow Mechanics](../tickets/forward-inverse-crypto-parity-mechanics.md) — Solved in coin space via linear regression $\Delta_{\text{btc}} = D_{\text{coin}} - (D_{\text{coin}} / F) K$ producing $F^* = -\alpha / \beta$ directly without spot dependency or circularity. Synthetic bounds via $F_{\text{bid}} = K / (1 - (C_{\text{ask}} - P_{\text{bid}}))$.
- [American Option Strike Filtering & Early Exercise Bounds](../tickets/forward-american-strike-filtering-and-bounds.md) — Model-free strike corridor $|\ln(K/S)| \le \kappa \sigma_{\text{atm}}\sqrt{T}$ where early exercise premium is within spread noise, combined with Merton analytical no-arbitrage bounds $[F_{\min}, F_{\max}]$ and escrowed dividend PV adjustment. **Amended 2026-10-25:** the rejection of PDE/lattice inversion is reopened; this corridor path stays as the fast, model-free tier unless [De-Americanization Method & Tier Structure](../tickets/forward-deamericanization-method-and-tiering.md) decides otherwise.
- [Arbitrage Diagnostics & Crossed-Quote Reporting](../tickets/forward-arbitrage-diagnostics-and-reporting.md) — ForwardDiagnostics and ForwardStatus (Valid, CrossedQuotes, BoxSpreadArbitrage, SpotDeviationExceeded). Graceful degradation returning results with status flags; two-pass outlier pruning with $3.5\times\text{RMSE}$ cutoff.
- [De-Americanization Method & Tier Structure](../tickets/forward-deamericanization-method-and-tiering.md) — Decided 2026-10-25 (grilling round 1). De-Americanize each quote to European-equivalent bid/ask under a trial carry (per-quote American IV), feed the existing `ImpliedForward` back-end, close the loop on carry, report the call/put IV gap as a diagnostic. Model-based default tier with the corridor as flagged fallback. Fast level = Ju (BAW alt.) closed-form; Accurate level = closed-form solve plus one CRR/FD premium correction, pending a prototype. Config enum with automatic fall back to Fast. Strike set: wider corridor, kappa set after fixtures.
- [American Carry, Dividend & Rate Inputs](../tickets/forward-american-carry-dividend-and-rate-inputs.md) — Decided 2026-10-26 (grilling rounds 1 and 2). Model path requires a caller-supplied discount factor (None runs the flagged corridor tier); optional discrete cash-dividend schedule used only by the FD correction pass of the Accurate level (FD spot model, fixed), ignored with a flag at Fast; a schedule makes Accurate the default; carry search [-0.10, 0.25] and flag range [-0.05, 0.15] are config fields. Evidence: flat carry costs 4 to 14 bp on regular dividends and about 45 bp on a special one; FD with the schedule stays under 2 bp.
- [American Solver API, Diagnostics & Fallback](../tickets/forward-american-solver-api-diagnostics-and-fallback.md) — Decided 2026-10-26 (grilling rounds 1 and 2). One Rust entry point with a nested `AmericanModelConfig`; orthogonal `tier`, `flags` bitset and numeric diagnostics, `ForwardStatus` gains only `WarningImplausibleCarry` (amended 2026-10-27: also `WarningModelFallback` and a `model_failure` cause, see the ticket); optional `atm_vol` estimated from quotes when absent; deterministic `max_accurate_prices` budget with no clock in the core; correction at mid, checked against per-side in fixtures. FFI: additive `_with_options` function and extended result, ABI stays 1. Python: extend `implied_forward`. Go: `ImpliedForwardWithOptions` and typed `ForwardConvention`, with the existing Go status/convention defects fixed first.
- [Hybrid Premium-Correction Prototype](../tickets/forward-hybrid-premium-correction-prototype.md) — Verdict 2026-10-26: yes. A Ju solve plus one CRR 400 correction pass matches the full CRR/FD solve (worst 0.36 bp, 0.21 bp against an independent CRR 4000 reference) at 8 to 12 ms instead of 260 to 520 ms; with a dividend schedule an FD-with-schedule correction turns up to 45 bp of flat-carry error into 1.1 bp in one pass and 0.8 bp in two. The Accurate level is feasible.
- [American Pricer Inventory, Accuracy & Latency Budget](../tickets/forward-american-pricer-inventory-and-latency.md) — Measured 2026-10-25. Ju/BAW closed forms: 1 to 2 us per price, forward recovered to about 6 bp; CRR 400 / FD 100x200: about 2 ms per price, under 2 bp, 0.3 to 0.9 s per IV-matching solve; only FD takes cash dividends; no vanilla `implied_volatility` exists yet; current parity/corridor path off by up to 114 bp (2y, 5% rates).
- [Rust Core API, Structs, and Trait Representation](../tickets/forward-rust-api-and-type-system.md) — Placed in `crates/libitofin/src/termstructures/forward/implied_forward.rs`. Exposes `OptionQuotePair`, `MarketConvention`, `ImpliedForwardConfig`, `ImpliedForwardResult`, and `ImpliedForward::calculate(...)`.
- [C-FFI, PyO3 Python Bindings, and Downstream Feed Migration](../tickets/forward-ffi-python-and-feed-migration.md) — Bound in C-FFI via `itofin_implied_forward_calculate` and in PyO3 via `itofin.termstructures.implied_forward`. Migrated `spx_option_iv/feed.py` and `btc_option_iv/market.py` with zero latency regressions.
- **Implementation delivered 2026-10-26** on branch `feat/american-implied-forward` (8 commits from `0b72b219` to `60b87a52`): Go status fixes, types, pricer adapter, Fast level, Accurate level, corridor and ATM-vol estimation, Python, FFI and Go, and 15 synthetic validation chains. Over those chains the forward error is at most 114 bp for plain parity, 6.0 bp at the Fast level without dividends and 0.82 bp at the Accurate level; the Accurate synthetic interval contains the true forward in all 15. The three items that were left open are settled: the mid-applied correction differs from a per-side one by at most 0.024 of a half spread; the corridor width `corridor_kappa` does not move the estimate between 0.25 and 3.0 on noise-free chains, so 1.0 stays; the work budget default became 100 because 64 flagged an ordinary 17-strike chain with a schedule.
- **Review and merge, 2026-10-26 to 2026-10-27** in four pull requests on `zhangys2/libitofin`: Go status, `IsCrossed` and carry fixes (#278); the model-based American forward with Fast and Accurate tiers across Rust, FFI, Python and Go (#279); two CI test fixes and the macOS Go leg dropped (#280); the failure cause, the `pairs_dropped` flag, a fixed strike set in the carry solve and noisy-quote validation (#281). Changes to the earlier decisions are in the [solver API ticket](../tickets/forward-american-solver-api-diagnostics-and-fallback.md): a `WarningModelFallback` status (code 7), a `model_failure` cause (codes 1 to 8), no spot gate on the model path, a dividend-shifted carry search.

## Not yet specified

- Public-surface decisions left over from the review: `#[non_exhaustive]` on `ForwardStatus` and the config structs (breaks struct-literal construction), carrying the pricer's error text through the result, and whether to close the 1e-9 window in the carry bracket search.
- Term structure of the borrow/carry rate across expiries for American underlyings (relates to the multi-expiry item below).
- Multi-expiry joint term-structure forward curve smoothing enforcing forward calendar monotonicity ($\partial F / \partial T$ consistency with interest rate / dividend term structures).
- Online streaming recursive Kalman filter state estimation for continuous implied forward tracking.
- Order book depth weighting incorporating Level 2 bid/ask volumes instead of top-of-book quotes alone.

## Out of scope

- Parametric stochastic volatility calibration (SABR, SVI, Heston) — the forward is a model-free input to volatility modeling.
- Full joint calibration of a stochastic-volatility model to American quotes in order to extract $F$; only de-Americanization with a pricer under the smile's own implied vol is in scope. (Replaces the earlier exclusion of all numerical de-Americanization, reopened 2026-10-25.)
- Live execution and order routing of arbitrage strategies (library provides analytics and diagnostics, not an execution engine).

## Frontier (open, unblocked, unclaimed)

*(None. Every ticket is closed; the remaining items are in Not yet specified. Implementation follows from the decisions above.)*


## Blocked (not frontier)

*(None)*

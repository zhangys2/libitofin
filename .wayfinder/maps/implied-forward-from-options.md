# Implied Forward Price & Discount Factor Engine

## Destination

Design and implement a unified implied forward price and discount factor estimation engine (`ImpliedForward`) in `libitofin` that calculates synthetic forward bid-ask intervals $[F_{\text{bid}}, F_{\text{ask}}]$, weighted point estimates $F^*$, and arbitrage diagnostics from option quotes with bid-ask spreads across European, American (model-free bounds / ATM strike filtering), and crypto-inverse markets. The engine must be implemented in Rust core (`crates/libitofin`), exposed via C-FFI (`crates/libitofin-ffi`) and Python (`crates/itofin-py`), validated against analytical fixtures and synthetic arbitrage scenarios, and integrated into the live SPX and BTC feeds in `example/btc-option-iv`.

## Notes

- **Domain**: Equity index (SPX), single stock/ETF (American options), cryptocurrency options (Deribit inverse BTC/ETH, cash-settled linear).
- **Core Principles**:
  - European Put-Call Parity: $C(K) - P(K) = D(T) \cdot (F - K)$.
  - Conversion / Reversal Arbitrage Bounds: $F_{\text{bid}} = K + (C_{\text{bid}} - P_{\text{ask}})/D$, $F_{\text{ask}} = K + (C_{\text{ask}} - P_{\text{bid}})/D$.
  - Weighted Least Squares (WLS): Weights $w_i = 1 / (\Delta C_i^2 + \Delta P_i^2)$ to suppress illiquid wing noise.
  - Model-Free American Bounds: Narrow-band ATM strike restriction where early exercise premium is negligible, plus analytical Merton bounds.
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
- **Market & Exercise Scope (2026-10-07 via Grilling Round 1)**: Tiered model-free approach — exact parity & WLS for European and Inverse Crypto; strike-restricted parity (filtering ITM strikes where early exercise is non-zero) + analytical early exercise bounds $[F_{\min}, F_{\max}]$ for American options.
- **Bid-Ask Spread Modeling (2026-10-07 via Grilling Round 1)**: Dual contract providing both synthetic forward spread $[F_{\text{bid}}, F_{\text{ask}}]$ from conversion/reversal bounds and weighted point estimate $F^*$ via WLS / robust median with weights $1/(\Delta C_i^2 + \Delta P_i^2)$.
- **Discount Factor Handling (2026-10-07 via Grilling Round 1)**: Hybrid mode — solve for $F$ with fixed $D$ if discount factor/curve provided; jointly solve $(F, D)$ via cross-strike regression if omitted.
- **Architectural Placement (2026-10-07 via Grilling Round 1)**: Core Rust in `crates/libitofin`, bound in `libitofin-ffi` and `crates/itofin-py`, consumed by example feeds.
- [Implied Forward Mathematical Formulation & WLS Objective](../tickets/forward-math-wls-and-bounds-formulation.md) — Closed-form WLS regression with spread weights $w_i = 1 / (\Delta C_i^2 + \Delta P_i^2 + \epsilon^2)$ estimating $(F^*, D^*)$ simultaneously via $y = D \cdot F - D \cdot K$. Strict + robust synthetic bounds $[F_{\text{bid}}, F_{\text{ask}}]$ per strike.
- [Inverse Crypto Option Parity & Cash Flow Mechanics](../tickets/forward-inverse-crypto-parity-mechanics.md) — Solved in coin space via linear regression $\Delta_{\text{btc}} = D_{\text{coin}} - (D_{\text{coin}} / F) K$ producing $F^* = -\alpha / \beta$ directly without spot dependency or circularity. Synthetic bounds via $F_{\text{bid}} = K / (1 - (C_{\text{ask}} - P_{\text{bid}}))$.
- [American Option Strike Filtering & Early Exercise Bounds](../tickets/forward-american-strike-filtering-and-bounds.md) — Model-free strike corridor $|\ln(K/S)| \le \kappa \sigma_{\text{atm}}\sqrt{T}$ where early exercise premium is within spread noise, combined with Merton analytical no-arbitrage bounds $[F_{\min}, F_{\max}]$ and escrowed dividend PV adjustment.
- [Arbitrage Diagnostics & Crossed-Quote Reporting](../tickets/forward-arbitrage-diagnostics-and-reporting.md) — ForwardDiagnostics and ForwardStatus (Valid, CrossedQuotes, BoxSpreadArbitrage, SpotDeviationExceeded). Graceful degradation returning results with status flags; two-pass outlier pruning with $3.5\times\text{RMSE}$ cutoff.
- [Rust Core API, Structs, and Trait Representation](../tickets/forward-rust-api-and-type-system.md) — Placed in `crates/libitofin/src/termstructures/forward/implied_forward.rs`. Exposes `OptionQuotePair`, `MarketConvention`, `ImpliedForwardConfig`, `ImpliedForwardResult`, and `ImpliedForward::calculate(...)`.
- [C-FFI, PyO3 Python Bindings, and Downstream Feed Migration](../tickets/forward-ffi-python-and-feed-migration.md) — Bound in C-FFI via `itofin_implied_forward_calculate` and in PyO3 via `itofin.termstructures.implied_forward`. Migrated `spx_option_iv/feed.py` and `btc_option_iv/market.py` with zero latency regressions.

## Not yet specified

- Multi-expiry joint term-structure forward curve smoothing enforcing forward calendar monotonicity ($\partial F / \partial T$ consistency with interest rate / dividend term structures).
- Online streaming recursive Kalman filter state estimation for continuous implied forward tracking.
- Order book depth weighting incorporating Level 2 bid/ask volumes instead of top-of-book quotes alone.

## Out of scope

- Parametric stochastic volatility calibration (SABR, SVI, Heston) — the forward is a model-free input to volatility modeling.
- Full numerical de-Americanization via iterative PDE/lattice solver inversion (rejected in favor of model-free strike-restricted bounds).
- Live execution and order routing of arbitrage strategies (library provides analytics and diagnostics, not an execution engine).

## Frontier (open, unblocked, unclaimed)

*(None — map destination achieved, ready for implementation)*

## Blocked (not frontier)

*(None)*

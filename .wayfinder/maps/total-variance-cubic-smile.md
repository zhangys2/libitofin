# Total Variance Cubic Smile & Roger Lee No-Arbitrage Wings

## Destination

Define the architecture, mathematical specification, wing asymptotics, and butterfly no-arbitrage constraints for a total-variance cubic smile section (`TotalVarianceCubicSmileSection`) in `libitofin` that:
1. Interpolates total implied variance $w(k) = \sigma^2(k) T$ against log-moneyness $k = \ln(K/F)$.
2. Enforces Roger Lee's extreme-strike asymptotic slope bounds ($\beta_R \in [0, 2]$, $\beta_L \in [-2, 0]$) to eliminate extreme-strike arbitrage via an exterior Smoothstep $C^2$ Transition Bridge without distorting interior fit.
3. Diagnoses and regularizes against butterfly arbitrage ($g(k) \ge 0$, Durrleman's condition) with an adaptive $\lambda$-ramp loop ($K_{\max} = 5$).
4. Implements the `SmileSection` trait for 1D single-expiry pricing.

All mathematical specifications, wing splice formulas, no-arbitrage verification mechanics, Rust core implementation, C-FFI, PyO3 Python bindings, and Go SDK wrappers are fully implemented, verified, and closed.

## Notes

- Domain: Equity & Crypto option implied volatility term structures.
- Foundation: Roger Lee (2004) *"The Moment Formula for Implied Volatility at Extreme Strikes"*; Durrleman (2003) condition for absence of butterfly arbitrage.
- Related Modules:
  - [`crates/libitofin/src/termstructures/volatility/total_variance_cubic_smile.rs`](../../crates/libitofin/src/termstructures/volatility/total_variance_cubic_smile.rs) (Rust core)
  - [`crates/itofin-py/src/cubicsmile.rs`](../../crates/itofin-py/src/cubicsmile.rs) (PyO3 Python bindings)
  - [`crates/libitofin-ffi/src/cubicsmile_api.rs`](../../crates/libitofin-ffi/src/cubicsmile_api.rs) (C-FFI)
  - [`sdk/go/total_variance_cubic_smile.go`](../../sdk/go/total_variance_cubic_smile.go) (Go SDK)
- Skills: `grilling` for HITL design decisions, `wayfinder` for decision ticket mapping, `tdd` for test-driven implementation.

## Decisions so far

- [Type structure & 1D slice scope](../tickets/tvcs-type-and-coordinate-definition.md) — Dedicated type `TotalVarianceCubicSmileSection` implementing `SmileSection` on 1D single-expiry slice first; coordinate is log-moneyness $k = \ln(K/F)$, ordinate is total variance $w = \sigma^2 T$.
- [Roger Lee wing enforcement & Smoothstep C2 bridge](../tickets/tvcs-roger-lee-wing-spec.md) — Spliced asymptotic linear wings with slope clamping to $\beta_R \in [0, 2 - \epsilon]$ and $\beta_L \in [-2 + \epsilon, 0]$. Spliced via an exterior Smoothstep $C^2$ Transition Bridge ($w'(u) = w'_b + (\beta - w'_b)(3u^2 - 2u^3)$) over width $\Delta k = \max(\Delta k_{\text{adjacent}}, \sigma_{\text{atm}}\sqrt{T})$. Maintains $C^2$ continuity everywhere ($w''(k_b) = 0$) and leaves interior regularized least-squares fit 100% untouched.
- [Least-squares QR solver & IV-normalized residual](../tickets/tvcs-least-squares-regularizer-formulation.md) — Adopted IV-normalized linear residual $\frac{w(k_i) - w_i}{2 \sigma_i T}$ with target $\frac{\sigma_i}{2}$ and dimensional curvature scaling $\lambda_{\text{eff}} = \lambda \frac{\sigma_{\text{atm}}}{4 \sqrt{T}}$ in $[1/\text{year}]$. Exact 2-point Gauss-Legendre quadrature integration with zero truncation error. Affine nullspace requires $N \ge 2$ distinct observations for full column rank QR solution in $< 1.5\,\mu\text{s}$.
- [Butterfly density diagnostic & adaptive regularizer ramp](../tickets/tvcs-butterfly-density-diagnostic.md) — Durrleman density $g(k) \ge 0$ evaluated across knots + 5-point interior Gauss-Legendre nodes per segment using segment polynomial coefficients with tolerance $\tau = 10^{-8}$. Proved $\lim_{|k|\to\infty} g(k) = (4 - \beta^2)/16 \ge 0$ under Lee bounds. Adaptive $\lambda \leftarrow 2\lambda$ ramp ($K_{\max} = 5$, $< 50\,\mu\text{s}$) with structured fallback policy (`AllowWithReport`).
- [FFI, Python bindings, and Go SDK surface](../tickets/tvcs-binding-and-ffi-surface.md) — Complete implementation across Rust core, C-FFI, PyO3 Python bindings, and Go SDK with zero clippy warnings and passing tests across all layers.

## Not yet specified

- Multi-expiry 2D volatility surface construction with calendar spread monotonicity ($\partial w / \partial T \ge 0$).
- Real-time Kalman state representation operating in total variance space.
- Calibration to raw bid/ask market quote prices rather than mid-IV observations.

## Out of scope

- Stochastic volatility parametric models (SABR, Heston, SVI).
- Negative interest rates or shifted lognormal variance adjustments ($F > 0$ strictly required).
- Full 2D surface construction (deferred to a separate downstream Wayfinder map).

## Frontier (open, unblocked, unclaimed)

*(None — map destination achieved)*

## Blocked (not frontier)

*(None)*
